#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>
#include <stdio.h>

#define ceil_div(n, d) ((n + d - 1) / (d))

// Shape-specialized GEMM paths for different workloads, not successive upgrades:
//   gemm_gemv_kernel    - M=1: vector loads and FP32 FMA/reduction compute only
//                         the real output row, without padded Tensor Core rows.
//   gemm_mma_llm_kernel - multi-row decoding and larger eligible matrices:
//                         shape-specific Tensor Core tiles and async staging.
// The launcher selects between them by shape. Both share the private namespace
// and helpers below with the progressive GEMM implementations.

namespace {
namespace wmma = nvcuda::wmma;

constexpr int TILE = 16;
constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_WARP_MASK = 0xffffffffu;

// Small-M masking threshold and common MMA pipeline dimensions.
constexpr int SMALL_M_LIMIT = 64;
constexpr int PIPE_K = 32;
constexpr int PIPE_THREADS = 4 * WARP_SIZE;

// Issue one nonblocking global->shared transfer of 16 bytes (eight FP16 values).
// This transfers raw bits directly into shared memory, without an intermediate
// register holding the data. Address and source-size operands still use registers.
// Caller contract:
//   1. dst points to a writable, 16-byte-aligned shared-memory region;
//   2. src is a 16-byte-aligned global pointer, with 16 readable bytes when valid=true.
//   3. The caller must not read/overwrite dst, or modify the copied source bytes,
//      until the appropriate async-copy wait has completed.
__device__ __forceinline__ void gemm_copy_async_16(half* dst, const half* src, bool valid) {
    // C++ passes a generic pointer. The PTX .shared destination instead needs
    // a shared-state-space address: convert first, then use its 32-bit offset.
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));

    // With valid=true, copy all eight halfs.
    // With valid=false, read zero source bytes and asynchronously zero-fill ALL
    // 16 dst bytes. Note zero-fill is necessary to init smem even source
    // address is out-of-bound.
    const int bytes = valid ? 16 : 0;

    // .cg requests L2 caching without L1 caching
    // .shared.global identify the destination/source address spaces.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(address), "l"(src), "r"(bytes)
                 : "memory");
}

// Close the calling thread's current batch: all its previously issued,
// uncommitted cp.async operations become one new group.
// An empty batch would create a trivially complete group.
__device__ __forceinline__ void gemm_copy_async_commit() {
    asm volatile("cp.async.commit_group;" : : : "memory");
}

// Wait until all committed groups older than the newest Pending groups have
// completed for THIS thread. The newest Pending groups may still be in flight;
template <int Pending> __device__ __forceinline__ void gemm_copy_async_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(Pending) : "memory");
}

// Explicit Tensor Core fragments allow a permuted shared layout, unlike the
// fixed row-major layout accepted by wmma::load_matrix_sync.
__device__ __forceinline__ void gemm_mma_load_a(unsigned (&fragment)[4], const half* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    // Each group of eight lanes supplies the row starts of one 8x8 matrix.
    // Four such matrices assemble a row-major 16x16 A operand in four registers.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "r"(address)
                 : "memory");
}

__device__ __forceinline__ void gemm_mma_load_b(unsigned (&fragment)[4], const half* ptr) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    // Transpose the four 8x8 pieces during the shared->register load: the global
    // and shared B matrices remain row-major, but mma.sync receives .col B.
    // Registers [0,1] serve the first eight output columns; [2,3] the next eight.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "r"(address)
                 : "memory");
}

__device__ __forceinline__ void gemm_mma_16x8(float (&acc)[4], const unsigned (&a)[4],
                                              const unsigned* b) {
    // One warp computes 16x8x16 with FP16 operands and FP32 accumulation.
    // Every lane owns four FP32 outputs. "+f" makes acc both input and output;
    // the packed FP16 operand registers use "r", not float arithmetic registers.
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Single-row GEMV: vectorized/coalesced B loads; FP32 accumulation and reduction.
// A[1,K] * B[K,N] -> C[1,N]: tile along N and split the K reduction within a CTA.
//   Vec:    adjacent output columns per lane; each warp spans 32 * Vec columns.
//   Warps:  warps cooperating on the SAME output columns, with disjoint K work.
//   Unroll: consecutive K elements per warp in each outer-loop iteration.
// Launch contract: blockDim.x = 32 * Warps, gridDim.x = N / (32 * Vec), and N must
// contain only full column tiles; the launcher checks this (no N bounds mask).
// Example <8,16,4>: 512 threads/CTA compute 256 unique outputs, not 16 separate
// 256-column tiles. Each output has 16 warp partials that are combined below.
template <int Vec, int Warps, int Unroll, bool CacheGlobal = true>
__global__ void gemm_gemv_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                 half* __restrict__ C, int N, int K, float alpha, float beta) {
    constexpr int Columns = 32 * Vec;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    // CTA b owns C[b * Columns : (b + 1) * Columns]. Within every warp, lane l owns
    // columns b * Columns + l * Vec + q, q=0..Vec-1. Different lanes never overlap;
    // the same lane in different warps contributes to the same output columns.
    const int col = blockIdx.x * Columns + lane * Vec;
    // Each thread keeps Vec partial dot products in registers. Unlike WMMA,
    // this computes only the one real output row, without 15 padded rows.
    float acc[Vec] = {};
    // For outer iteration t, warp w handles k = t * (Warps * Unroll) + w * Unroll + u.
    // With Warps = 16, Unroll = 4: warp 0 handles 0..3, warp 1 handles 4..7, ...,
    // warp 15 handles 60..63; next iteration warp 0 handles 64..67, etc.
    // Together the warps cover K exactly once per output; k<K masks a K tail.
    for (int base = warp * Unroll; base < K; base += Warps * Unroll) {
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int k = base + u;
            if (k < K) {
                // All lanes in this warp use the same A[k]; each lane reuses it
                // across its Vec columns. B is row-major: lane vectors occupy
                // adjacent portions of the same B[k, CTA-column-tile] segment.
                const float a = __half2float(A[k]);
                const half* bp = B + size_t(k) * N + col;
                // Packed loads only move FP16 bits. Convert operands to FP32
                // before multiplying; there is no half/half2 accumulation.
                unsigned bits[Vec / 2];
                // Perf experiment:
                // RTX PRO 1000/SM120, CUDA 13.2: <8,16,4>, M/N/K=1/2048/1024.
                // C++ -> PTX .cg: 36.176 -> 25.872 us cold (-28.5%), 20.320 -> 15.552 us warm (-23.5%).
                // Main cause here is compiler-selected 2x OUTER-loop unrolling for ptx version.
                // The .cg cache hint only has 1% perf improvement on controlled experiment.
                // smsp__inst_executed.sum: 270216 -> 226832; global-load instructions stay 16384,
                // dram__bytes_op_read.sum: 4211200 -> 4215040. Counts alone do not explain the speedup.
                // Controls: kernel replay, base clocks, cache all/none, 20 warmups + 20 samples/variant.
                if constexpr (CacheGlobal) {
                    if constexpr (Vec == 2)
                        asm volatile("ld.global.cg.u32 %0, [%1];"
                                     : "=r"(bits[0])
                                     : "l"(bp)
                                     : "memory");
                    if constexpr (Vec == 4)
                        asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                                     : "=r"(bits[0]), "=r"(bits[1])
                                     : "l"(bp)
                                     : "memory");
                    if constexpr (Vec == 8)
                        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3])
                                     : "l"(bp)
                                     : "memory");
                } else {
                    if constexpr (Vec == 2)
                        bits[0] = *reinterpret_cast<const unsigned*>(bp);
                    if constexpr (Vec == 4) {
                        const uint2 p = *reinterpret_cast<const uint2*>(bp);
                        bits[0] = p.x;
                        bits[1] = p.y;
                    }
                    if constexpr (Vec == 8) {
                        const uint4 p = *reinterpret_cast<const uint4*>(bp);
                        bits[0] = p.x;
                        bits[1] = p.y;
                        bits[2] = p.z;
                        bits[3] = p.w;
                    }
                }
#pragma unroll
                for (int q = 0; q < Vec / 2; ++q) {
                    __half2_raw raw;
                    raw.x = static_cast<unsigned short>(bits[q]);
                    raw.y = static_cast<unsigned short>(bits[q] >> 16);
                    const float2 b = __half22float2(static_cast<half2>(raw));
                    acc[2 * q] = fmaf(a, b.x, acc[2 * q]);
                    acc[2 * q + 1] = fmaf(a, b.y, acc[2 * q + 1]);
                }
            }
        }
    }
    // partial[w][q][l] holds warp w's K-partial for output column
    // blockIdx.x*Columns + l*Vec + q. SoA keeps lanes contiguous for each q,
    // giving each scalar warp-wide access 32 distinct shared banks.
    // Synchronize the whole CTA so warp 0 can read every warp's partials.
    __shared__ float partial[Warps][Vec][32];
#pragma unroll
    for (int q = 0; q < Vec; ++q) {
        partial[warp][q][lane] = acc[q];
    }
    __syncthreads();
    // Reduce across WARPS: lane l of warp 0 sums the Warps partials
    // for each of its Vec outputs, then writes alpha*dot + beta*C_initial.
    if (warp == 0) {
#pragma unroll
        for (int q = 0; q < Vec; ++q) {
            float sum = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                sum += partial[w][q][lane];
            }
            const float previous = beta != 0 ? __half2float(C[col + q]) : 0.f;
            C[col + q] = __float2half_rn(fmaf(alpha, sum, beta * previous));
        }
    }
}

// Multi-row workload: reuse staged operands across Tensor Core output tiles;
// this complements the single-row GEMV path rather than replacing it.
// Swizzled Tensor Core GEMM for LLM-shaped tiles. All variants use a 1D grid;
// GroupRows controls tile rasterization. Sub-64-row tiles mask partial M rows;
// larger tiles require full M tiles. Each CTA reduces over the complete K axis.
//
// The diagrams use <64,64,Stages,1> as an illustrative case only.
// The dispatcher does not launch a 64x64 MMA specialization.
//
// 64x64 illustrative output tiling hierarchy (M = rows, N = columns, K = reduction dimension):
//
//   Grid: (N/64)*(M/64) blocks, with 128 threads = 4 warps per CTA.
//   CTA tile_id=blockIdx.x computes one 64x64 C tile:
//     by = tile_id/(N/64), bx = tile_id%(N/64)
//     tile_row = 64*by, tile_col = 64*bx
//     C rows [tile_row..tile_row+63], columns [tile_col..tile_col+63].
//   This CTA reduces over ALL K; different CTAs do not split the K reduction.
//
//   Within one CTA's output tile (row/column labels are CTA-local):
//
//                         N columns
//                    0..31             32..63
//              +------------------+------------------+
//    M   0..31 | warp 0: 32x32    | warp 1: 32x32    |
//   rows       | threads 0..31    | threads 32..63   |
//              +------------------+------------------+
//       32..63 | warp 2: 32x32    | warp 3: 32x32    |
//              | threads 64..95   | threads 96..127  |
//              +------------------+------------------+
//
//   warp = threadIdx.x/32
//   warp_row = (warp/2)*32, warp_col = (warp%2)*32
//   Global output origin = (tile_row+warp_row, tile_col+warp_col).
//   Warps partition M/N, NOT K: all four process the same sequence of K tiles.
//
//   Within each warp's 32x32 output (labels are relative to that warp):
//
//                          N columns
//                    0..7    8..15   16..23   24..31
//                 +--------+--------+--------+--------+
//    M    0..15   | (0,0)  | (0,1)  | (0,2)  | (0,3)  |
//                 +--------+--------+--------+--------+
//        16..31   | (1,0)  | (1,1)  | (1,2)  | (1,3)  |
//                 +--------+--------+--------+--------+
//
//   Box (i,j) is a 16x8 output tile held in acc[i][j][0..3] PER LANE.
//   One warp-wide mma.m16n8k16 updates one box from a K=16 slice:
//     A: 16x16  @  B: 16x8  ->  16x8 accumulated outputs.
//   Thus compute_group() issues 2 M boxes x 4 N boxes = 8 MMAs per K=16.
//   Each shared K=32 tile supplies two such groups, updating the SAME outputs;
//   repeat for K/32 tiles, keeping the FP32 accumulators live throughout.
template <int BlockM, int BlockN, int Stages, int GroupRows = 1>
__global__ void gemm_mma_llm_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                    float alpha, float beta) {
    // Only the 16/32-row small-M tiles may contain a partial final row tile.
    // Dispatch guarantees full M tiles for BlockM >= SMALL_M_LIMIT.
    constexpr bool MaskM = BlockM < SMALL_M_LIMIT;
    constexpr int WarpRows = BlockM == 16 ? 1 : 2; // each warp row has 16 M rows, same size as M fragment
    constexpr int WarpCols = 4 / WarpRows; // 4 warps per block
    constexpr int MFrags = BlockM / WarpRows / 16;
    constexpr int NFrags = BlockN / WarpCols / 16;
    static_assert(BlockM % (16 * WarpRows) == 0 && BlockN % (16 * WarpCols) == 0);
    static_assert(Stages >= 2 && Stages <= 6, "Use two to six async-copy stages");
    union SharedStorage {
        struct {
            half a[Stages][BlockM][PIPE_K];
            half b[Stages][PIPE_K][BlockN];
        } input;
        // +4 FP32 columns change bank mapping while preserving 16-byte row alignment.
        // In the 64x64 example: 68 floats/row = 272 bytes (still 16-byte aligned).
        float product[BlockM][(BlockN + 4)];
    };
    __shared__ __align__(32) SharedStorage shared;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    // Four warps cover the CTA. Each warp has an MFrags-by-NFrags register
    // tile, so larger CTA tiles reuse each loaded operand for more MMAs.
    const int warp_row = (warp / WarpCols) * (BlockM / WarpRows);
    const int warp_col = (warp % WarpCols) * (BlockN / WarpCols);
    const int n_tiles = N / BlockN; // launcher makes sure N is divisible by BlockN to make vector load simpler
    int tile_row, tile_col;
    if constexpr (GroupRows == 1) {
        tile_row = (blockIdx.x / n_tiles) * BlockM;
        tile_col = (blockIdx.x % n_tiles) * BlockN;
    } else {
        // Rasterize a group of output-row tiles before advancing columns.
        // Nearby CTAs reuse B in L2; bounded row groups also keep their A working
        // set manageable. The final group can contain fewer than GroupRows rows.
        //                     Tile column
        //                 0    1    2    3
        //               ┌────┬────┬────┬────┐
        // tile row 0    │  0 │  2 │  4 │  6 │ ┐
        //               ├────┼────┼────┼────┤ │ group 0
        // tile row 1    │  1 │  3 │  5 │  7 │ ┘
        //               ├────┼────┼────┼────┤
        // tile row 2    │  8 │ 10 │ 12 │ 14 │ ┐
        //               ├────┼────┼────┼────┤ │ group 1
        // tile row 3    │  9 │ 11 │ 13 │ 15 │ ┘
        //               ├────┼────┼────┼────┤
        // tile row 4    │ 16 │ 17 │ 18 │ 19 │ group 2: only one row
        //               └────┴────┴────┴────┘
        // Workload (M,N,K)	    GroupRows	Ungrouped (1)	Grouped	Result
        // QKV (2048,6144,4096)	4	        5.953 ms	    3.408 ms (42.75% less time)
        //
        // Note: increasing groups beyond 4 has no futher gain on the tested sm120 GPU.
        //
        // Stats show better cache reuse:
        // For QKV, GroupRows=4 changes:
        // - DRAM reads: 1549.7 → 64.0 MiB, a 95.9% reduction.
        // - L2 sector hit rate: 32.7% → 96.2%.
        const int m_tiles = (M + BlockM - 1) / BlockM;
        const int group_size = GroupRows * n_tiles;
        const int group = blockIdx.x / group_size, first_m = group * GroupRows;
        const int rows = min(GroupRows, m_tiles - first_m);
        const int within = blockIdx.x - group * group_size;
        tile_row = (first_m + within % rows) * BlockM;
        tile_col = (within / rows) * BlockN;
    }
    const int k_tiles = K / PIPE_K;

    // Logical shared tiles: A[BlockM][PIPE_K], B[PIPE_K][BlockN].
    // Threads copy eight-half (16-byte) vectors
    // XOR permutes whole vectors without transposing either stored matrix:
    //   A physical_col = logical_col ^ (((row >> 1) & 3) * 8)
    //   B physical_col = logical_col ^ ((row & 7) * 8)
    // Fragment loads below use the same permutations.
    //
    // 64x64 illustrative example; other tiles use the BlockM/BlockN loop bounds below.
    // Logical slices (global matrices are row-major; offsets count halfs):
    //   A[M,K]                           B[K,N]
    //     columns k0..k0+31                 columns tile_col..tile_col+63
    //     +----------------+               +--------------------------+
    //     | 64 rows x 32 K |               | 32 K rows x 64 columns   |
    //     +----------------+               +--------------------------+
    //      rows tile_row..tile_row+63        rows k0..k0+31
    //             | cp.async + XOR                  | cp.async + XOR
    //             v                                 v
    //      a[slot][64][32]                    b[slot][32][64]
    //
    // A[64×32]   K columns: 0 ─────────────────── 31
    //    M rows
    //      0–7   ┌─────────────────────────────────┐
    //            │ Warp 0                          │
    //      8–15  ├─────────────────────────────────│
    //            │ Warp 1                          │
    //     16–23  ├─────────────────────────────────┤
    //            │ Warp 2                          │
    //     24–31  ├─────────────────────────────────┤
    //            │ Warp 3                          │
    //     32–39  ├─────────────────────────────────┤
    //            │ Warp 0                          │
    //     40–47  ├─────────────────────────────────┤
    //            │ Warp 1                          │
    //     48–55  ├─────────────────────────────────┤
    //            │ Warp 2                          │
    //     56–63  ├─────────────────────────────────┤
    //            │ Warp 3                          │
    //            └─────────────────────────────────┘
    // B[32×64]              N columns: 0 ─────────────────── 63
    //    K rows
    //      0–3   ┌──────────────────────────────────────────────────┐
    //            │ Warp 0                                           │
    //      4–7   ├──────────────────────────────────────────────────┤
    //            │ Warp 1                                           │
    //      8–11  ├──────────────────────────────────────────────────┤
    //            │ Warp 2                                           │
    //     12–15  ├──────────────────────────────────────────────────┤
    //            │ Warp 3                                           │
    //     16–19  ├──────────────────────────────────────────────────┤
    //            │ Warp 0                                           │
    //     20–23  ├──────────────────────────────────────────────────┤
    //            │ Warp 1                                           │
    //     24–27  ├──────────────────────────────────────────────────┤
    //            │ Warp 2                                           │
    //     28–31  ├──────────────────────────────────────────────────┤
    //            │ Warp 3                                           │
    //            └──────────────────────────────────────────────────┘
    //
    // One vector Vj = eight consecutive FP16 values = 16 bytes
    //
    // LOGICAL vector -> PHYSICAL vector slot:
    //   A physical half-column start:  0   8  16  24
    //   row       0,1                [V0][V1][V2][V3]
    //             2,3                [V1][V0][V3][V2]
    //             4,5                [V2][V3][V0][V1]
    //             6,7                [V3][V2][V1][V0]
    //   destination vector = j XOR ((row >> 1) & 3)
    //
    //   B physical half-column start:  0   8  16  24  32  40  48  56
    //   row       0                  [V0][V1][V2][V3][V4][V5][V6][V7]
    //             1                  [V1][V0][V3][V2][V5][V4][V7][V6]
    //             2                  [V2][V3][V0][V1][V6][V7][V4][V5]
    //             3                  [V3][V2][V1][V0][V7][V6][V5][V4]
    //             4                  [V4][V5][V6][V7][V0][V1][V2][V3]
    //             5                  [V5][V4][V7][V6][V1][V0][V3][V2]
    //             6                  [V6][V7][V4][V5][V2][V3][V0][V1]
    //             7                  [V7][V6][V5][V4][V3][V2][V1][V0]
    //   destination vector = j XOR (row & 7)
    //
    // Each stage occupies 4096 bytes per operand.
    auto copy_stage = [&](int slot, int k_tile) {
        if (k_tile >= k_tiles) {
            // Keep the fixed wait-group distance without issuing unused copies.
            gemm_copy_async_commit();
            return;
        }
        const bool valid = k_tile < k_tiles;
#pragma unroll
        for (int v = tid; v < BlockM * PIPE_K / 8; v += PIPE_THREADS) {
            const int row = v / (PIPE_K / 8), col = (v % (PIPE_K / 8)) * 8;
            // Permute WHOLE eight-half vectors, never elements within a vector.
            // A has four vectors/row: row pairs select different vector columns.
            const int physical_col = col ^ (((row >> 1) & 3) * 8);
            // Padding is only along M: N and K are full tiles at dispatch.
            // Zero-fill missing A rows; all lanes still issue/commit their groups.
            const bool valid_a = valid && (!MaskM || tile_row + row < M);
            const half* src = valid_a ? A + (tile_row + row) * K + k_tile * PIPE_K + col : A;
            gemm_copy_async_16(&shared.input.a[slot][row][physical_col], src, valid_a);
        }
#pragma unroll
        for (int v = tid; v < PIPE_K * BlockN / 8; v += PIPE_THREADS) {
            const int row = v / (BlockN / 8), col = (v % (BlockN / 8)) * 8;
            // B permutes whole eight-half vectors: each of eight rows selects a different
            // 16-byte bank group. Both copy and ldmatrix must use the SAME XOR.
            const int physical_col = col ^ ((row & 7) * 8);
            const half* src = valid ? B + (k_tile * PIPE_K + row) * N + tile_col + col : B;
            gemm_copy_async_16(&shared.input.b[slot][row][physical_col], src, valid);
        }
        // Commit is per-thread and does not wait; publish with wait + CTA barrier below.
        gemm_copy_async_commit();
    };

    // load_fragments(buffer, slot, group): load A 32×16 and B 16×32 each time
    // per warp.
    // Note this means we need two loads (groups) to consume the total bytes
    // loaded from one copy_stage.
    //
    // Exmaple:
    // One CTA-wide copy_stage loads K = 0..31 into shared memory:

    //    A tile: 64 × 32                     B tile: 32 × 64
    //    rows 0..63, K 0..31                K 0..31, cols 0..63
    //              │                                  │
    //              └──────── split K into two groups ─┘
    //                            │
    //           ┌────────────────┴────────────────┐
    //           │                                 │
    //    group 0: K=0..15                  group 1: K=16..31
    //    A: 64×16, B: 16×64               A: 64×16, B: 16×64
    //           │                                 │
    //    4 warps load fragments           4 warps load fragments
    //    and update their C tiles         and update those same C tiles
    //
    // 64x64 illustrative example: MFrags=NFrags=2; each warp computes 32x32.
    // Full shared tiles are A[64][32] and B[32][64]. One warp uses 32 A rows
    // starting at warp_row and 32 B columns starting at warp_col:
    //
    //   A: warp's 32x32 logical view       B: warp's 32x32 logical view
    //             K=0..15   K=16..31                N=0..15   N=16..31
    //           +----------+----------+          +----------+----------+
    //   M=0..15 | A(0,0)   | A(0,1)   |  K=0..15 | B(0,0)   | B(0,1)   |
    //           +----------+----------+          +----------+----------+
    //  M=16..31 | A(1,0)   | A(1,1)   | K=16..31 | B(1,0)   | B(1,1)   |
    //           +----------+----------+          +----------+----------+
    //   M,N labels are relative to the warp; K labels are within this slot.
    //   Each box is 16x16. For g=group and b=buffer, the loads are:
    //
    //     A(i,g) -- ldmatrix.x4 -------> a_frag[b][i][0..3], i=0,1
    //     B(g,j) -- ldmatrix.x4.trans -> b_frag[b][j][0..3], j=0,1
    //
    // These are LOGICAL views, not contiguous physical shared rectangles.
    // Every 8-half row vector is found using the SAME XOR as copy_stage():
    //   A physical_col = logical_col ^ (((row >> 1) & 3) * 8)
    //   B physical_col = logical_col ^ ((row & 7) * 8)
    // The physical vector permutations are drawn above copy_stage().
    //
    // Each 16x16 box is loaded as four 8x8 pieces. Labels below are register
    // INDICES: the whole warp receives each piece, two halfs per lane/register.
    //
    //       A(i,g): local [M,K]              B(g,j): local [K,N]
    //             K=0..7  K=8..15                  N=0..7  N=8..15
    //            +-------+-------+               +-------+-------+
    //     M=0..7 | reg 0 | reg 2 |        K=0..7 | reg 0 | reg 2 |
    //            +-------+-------+               +-------+-------+
    //    M=8..15 | reg 1 | reg 3 |       K=8..15 | reg 1 | reg 3 |
    //            +-------+-------+               +-------+-------+
    //
    // Note: reg0 means all 32 lanes's reg0 to total holds M=0..7;k=0..7
    //       elements. each lane's reg0 hold 2 packed elements. 2 * 32 lanes
    //       hold total 64 MK elements
    //
    unsigned a_frag[2][MFrags][4], b_frag[2][NFrags][4]; // [register buffer][logical 16x16 tile][packed register]
    float acc[MFrags][2 * NFrags][4] = {}; // MFrags x (2*NFrags) m16n8 accumulators, 4 FP32 outputs/lane
    auto load_fragments = [&](int buffer, int slot, int group) {
#pragma unroll
        for (int i = 0; i < MFrags; ++i) {
            const int row = warp_row + i * TILE + lane % TILE;
            const int col = group * TILE + (lane / TILE) * 8;
            gemm_mma_load_a(a_frag[buffer][i],
                            &shared.input.a[slot][row][col ^ (((row >> 1) & 3) * 8)]);
        }
#pragma unroll
        for (int j = 0; j < NFrags; ++j) {
            const int row = group * TILE + lane % TILE;
            const int col = warp_col + j * TILE + (lane / TILE) * 8;
            gemm_mma_load_b(b_frag[buffer][j], &shared.input.b[slot][row][col ^ ((row & 7) * 8)]);
        }
    };
    // Each 16x16 B box feeds two N=8 MMAs: j/2 selects the box and
    // (j%2)*2 selects register pair [0,1] or [2,3]. One K=16 group updates
    // MFrags * (2*NFrags) accumulator boxes of shape 16x8.
    // 64x64 illustrative example; other shapes extend the i/j loops using MFrags/NFrags.
    //
    //                                    Output columns, relative to warp
    //                           0..7        8..15       16..23      24..31
    //                             j=0         j=1         j=2         j=3
    //                        +------------+------------+------------+------------+
    // a_frag[b][0]  i=0      | acc[0][0]  | acc[0][1]  | acc[0][2]  | acc[0][3]  |
    //  rows 0..15            |   16×8     |   16×8     |   16×8     |   16×8     |
    //                        +------------+------------+------------+------------+
    // a_frag[b][1]  i=1      | acc[1][0]  | acc[1][1]  | acc[1][2]  | acc[1][3]  |
    //  rows 16..31           |   16×8     |   16×8     |   16×8     |   16×8     |
    //                        +------------+------------+------------+------------+
    //
    //                             PER-LANE B registers
    // b_frag[b][0]:       [ reg 0, reg 1 ] [ reg 2, reg 3 ]
    //                          j=0              j=1
    //                       N=0..7           N=8..15
    // b_frag[b][1]:       [ reg 0, reg 1 ] [ reg 2, reg 3 ]
    //                          j=2              j=3
    //                      N=16..23         N=24..31
    //
    //                              Output columns
    // Rows             0,1  2,3  4,5  6,7 | 8,9 10,11 12,13 14,15
    // ────────────────────────────────────┼───────────────────────
    //  0, 8,16,24      L00  L01  L02  L03 | L00  L01   L02   L03
    //  1, 9,17,25      L04  L05  L06  L07 | L04  L05   L06   L07
    //  2,10,18,26      L08  L09  L10  L11 | L08  L09   L10   L11
    //  3,11,19,27      L12  L13  L14  L15 | L12  L13   L14   L15
    //  4,12,20,28      L16  L17  L18  L19 | L16  L17   L18   L19
    //  5,13,21,29      L20  L21  L22  L23 | L20  L21   L22   L23
    //  6,14,22,30      L24  L25  L26  L27 | L24  L25   L26   L27
    //  7,15,23,31      L28  L29  L30  L31 | L28  L29   L30   L31
    auto compute_group = [&](int buffer) {
#pragma unroll
        for (int i = 0; i < MFrags; ++i) {
#pragma unroll
            for (int j = 0; j < 2 * NFrags; ++j) {
                gemm_mma_16x8(acc[i][j], a_frag[buffer][i], &b_frag[buffer][j / 2][(j % 2) * 2]);
            }
        }
    };

// issue Stage - 1 copies, one slot is empty
#pragma unroll
    for (int slot = 0; slot < Stages - 1; ++slot)
        copy_stage(slot, slot);
    // Complete tile 0's copies for each thread. Newer copy groups may remain in flight.
    gemm_copy_async_wait<Stages - 2>();
    __syncthreads();
    // load first 16 element half
    load_fragments(0, 0, 0);

    // At loop entry, register buffer 0 holds group 0 (first 16 element half) of tile k_tile.
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        // start to fetch next Stage. Early returns if out-of-bound.
        // But we still commit even if early return to keep the
        // gemm_copy_async_wait consistent during the loop (that it assumes a
        // new stage has been prefetch in this loop).
        copy_stage((k_tile + Stages - 1) % Stages, k_tile + Stages - 1);

        // Load the current tile's second half (K offsets 15..32)
        // [frag 0 (ready), frag 1 (loading)]
        load_fragments(1, k_tile % Stages, 1);
        // Accumulate the current tile's first half (K offsets 0..15).
        compute_group(0);

        // Wait until this thread's copies for tile k_tile+1 are complete;
        // the newest Stages-2 commit groups may remain pending.
        gemm_copy_async_wait<Stages - 2>();
        __syncthreads();

        // Buffer 0's old operands were consumed above: prefetch group 0 of the
        // NEXT tile into it before computing the CURRENT tile's second half.
        if (k_tile + 1 < k_tiles) {
            load_fragments(0, (k_tile + 1) % Stages, 0);
        }
        // Accumulate the current tile's second half; buffer 0 is ready for
        // the next iteration (no more shared reads of the current tile).
        compute_group(1);
    }
    // Tail zero-fills still write shared memory. Drain every copy from every
    // lane before aliasing the input ring with the epilogue's output scratch.
    gemm_copy_async_wait<0>();
    __syncthreads();
    // R=warp_row+i*TILE, u=lane/4; Q=warp_col+j*8, v=2*(lane%4).
    //
    //                           shared columns
    //                       Q+v               Q+v+1
    //                    +-----------------+-----------------+
    //   shared row R+u   | acc[i][j][0]    | acc[i][j][1]    | <- float2, r=0
    //                    +-----------------+-----------------+
    //                       (7 intervening shared rows)
    //                    +-----------------+-----------------+
    //   shared row R+u+8 | acc[i][j][2]    | acc[i][j][3]    | <- float2, r=1
    //                    +-----------------+-----------------+
#pragma unroll
    for (int i = 0; i < MFrags; ++i) {
#pragma unroll
        for (int j = 0; j < 2 * NFrags; ++j) {
            // PTX m16n8k16 FP32 accumulator mapping (not an assumed WMMA mapping):
            // lane/4 owns row r and r+8; lane%4 owns two adjacent columns.
            const int row = warp_row + i * TILE + lane / 4;
            const int col = warp_col + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                *reinterpret_cast<float2*>(&shared.product[row + r * 8][col]) =
                    make_float2(acc[i][j][r * 2], acc[i][j][r * 2 + 1]);
            }
        }
    }
    // Reorder into contiguous 16-byte C transfers. Direct per-lane half2 stores
    // were tested but slower; the shared epilogue avoids their scattered requests.
    __syncthreads();
    for (int v = tid; v < BlockM * BlockN / 8; v += PIPE_THREADS) {
        const int row = v / (BlockN / 8), col = (v % (BlockN / 8)) * 8;
        if constexpr (MaskM) {
            if (tile_row + row >= M)
                continue;
        }
        const int index = (tile_row + row) * N + tile_col + col;
        __align__(16) half previous[8], output[8];
        if (beta != 0.0f) {
            *reinterpret_cast<float4*>(previous) = *reinterpret_cast<const float4*>(C + index);
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float value = beta != 0.0f ? __half2float(previous[q]) : 0.0f;
            output[q] = __float2half_rn(alpha * shared.product[row][col + q] + beta * value);
        }
        *reinterpret_cast<float4*>(C + index) = *reinterpret_cast<const float4*>(output);
    }
}

} // namespace
