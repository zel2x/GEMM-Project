#pragma once
#include "gemm_llm_kernels.cuh"

// Progressive GEMM implementations: start with scalar global-memory loads,
// add shared-memory reuse and wider loads, then use warp-level Tensor Cores.
//
// Global-memory scalar baseline:
//   gemm_base_kernel          - one output/thread; direct scalar FP16 loads.
// Shared-memory / FP32 FMA family:
//   gemm_smem_kernel          - reuse 32x32 input tiles across threads.
//   gemm_smem_reg_kernel      - keep a 2x2 output tile per thread in registers.
//   gemm_smem_reg_half2_kernel - pair FP16 global loads; same 2x2 output tile.
//   gemm_f4_kernel            - 16-byte global loads and eight outputs/thread;
//                               an alternative shared-memory/vector-load design.
// WMMA / Tensor Core family:
//   gemm_wmma_smem_kernel     - one warp/16x16 tile; stage and pad input edges.
//   gemm_wmma_gmem_kernel     - direct-global WMMA for full aligned tiles;
//                               an alternative to the staged boundary path.
//   gemm_wmma_smem_reg_kernel - reuse fragments across a 32x32 output tile.
//   gemm_wmma_pipeline_kernel - four warps/64x64 tile with cp.async staging.
//   gemm_wmma_swizzle_kernel  - XOR-swizzled shared layout and explicit
//                               ldmatrix/mma.sync; same 64x64 tile family.

// Basic GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C_initial[M,N].
// All matrices are row-major FP16; each dot product accumulates in FP32.
#define tile_side 32
#define reg_size 2

namespace {

// 1. Global-memory loads: scalar baseline.
// Scalar baseline: one output/thread, one FP16 value/global load
__global__ void gemm_base_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                                 float beta) {
    int col_id = threadIdx.x + blockDim.x * blockIdx.x;
    int row_id = threadIdx.y + blockDim.y * blockIdx.y;
    if (col_id >= N || row_id >= M) {
        return;
    }
    float acc = 0.0f;
    for (int k=0; k < K; k++) {
        float a = __half2float(A[row_id * K + k]);
        float b = __half2float(B[k * N + col_id]);
        acc = fmaf(a, b, acc);
    }

    float C_init = (beta != 0.0f) ? __half2float(C[row_id * N + col_id]) : 0.0f;
    C[row_id * N + col_id] = __float2half_rn(fmaf(alpha, acc, beta * C_init));
}

// 2. Shared-memory staging with ordinary FP32 fused multiply-add (FMA).
// Compared with the baseline, stage each input tile once for reuse by threads.
__global__ void gemm_smem_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                                 float beta) {
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int row_id = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float sA[tile_side][tile_side];
    __shared__ float sB[tile_side][tile_side];
    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += tile_side) {
        // load share memory to tile
        sA[threadIdx.y][threadIdx.x] = (row_id < M && k_base + threadIdx.x < K) ? __half2float(A[row_id * K + k_base + threadIdx.x]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col_id < N && k_base + threadIdx.y < K) ? __half2float(B[(k_base + threadIdx.y) * N + col_id]) : 0.0f;
        __syncthreads();

        for(int k = 0 ; k < tile_side; k++) {
            acc = fmaf(sA[threadIdx.y][k], sB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    // write sC back
    if (col_id < N && row_id < M) {
        float C_prev = (beta == 0.0f) ? 0.0f : __half2float(C[row_id * N + col_id]);
        C[row_id * N + col_id] = __float2half_rn(fmaf(acc, alpha, C_prev * beta));
    }
}

// Compared with gemm_smem_kernel, compute four outputs/thread (2x2) and
// reuse each register operand across two products. The shared tile stays 32x32.
__global__ void gemm_smem_reg_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                                     float beta) {
    int col_id = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
    int row_id = (threadIdx.y + blockDim.y * blockIdx.y) * 2;
    int local_row_id = threadIdx.y * 2;
    int local_col_id = threadIdx.x * 2;
    float acc[2][2] = {0.0f};
    // load tile
    __shared__ float sA[tile_side][tile_side];
    __shared__ float sB[tile_side][tile_side];

    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    for (int k_base = 0; k_base < K; k_base += tile_side) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int A_rowid = row_id + j;
                int A_colid = k_base + local_col_id + i;
                int B_rowid = k_base + local_row_id + j;
                int B_colid = col_id + i;
                sA[local_row_id + j][local_col_id + i] = (A_rowid < M && A_colid < K) ? __half2float(A[A_rowid * K + A_colid]) : 0.0f;
                sB[local_row_id + j][local_col_id + i] = (B_rowid < K && B_colid < N) ? __half2float(B[B_rowid * N + B_colid]) : 0.0f;
            }
        }
        __syncthreads();
        for (int k = 0; k < tile_side; k++) {
            float rA[2] = {0.0f};
            float rB[2] = {0.0f};
            for (int i = 0; i < 2; i++) {
                rA[i] = sA[local_row_id + i][k];
                rB[i] = sB[k][local_col_id + i];
            }
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    acc[j][i] = fmaf(rA[j], rB[i], acc[j][i]);
                }
            }
        }
        __syncthreads();
    }

    float C_prev = 0.0f;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            int C_rowid = row_id + j;
            int C_colid = col_id + i;
            if (C_rowid < M && C_colid < N) {
                C_prev = (beta != 0.0f) ? __half2float(C[C_rowid * N + C_colid]) : 0.0f;
                C[C_rowid * N + C_colid] = __float2half_rn(fmaf(alpha, acc[j][i], C_prev * beta));
            }
        }
    }
}

// A float-sized (32-bit) global load contains TWO FP16 values, not one FP32
// number. half2 preserves that interpretation and converts directly to float2.
__device__ __forceinline__ float2 load_gemm_half2(const half* matrix, int rows, int cols,
                                                int row, int col) {
    if (row >= rows || col >= cols) {
        return make_float2(0.0f, 0.0f);
    }
    const half* ptr = matrix + static_cast<size_t>(row) * cols + col;
    if (col + 1 < cols &&
        (reinterpret_cast<std::uintptr_t>(ptr) & (alignof(half2) - 1)) == 0) {
        return __half22float2(*reinterpret_cast<const half2*>(ptr));
    }
    // Odd leading dimensions, offset pointers, and the last column use scalars.
    return make_float2(__half2float(ptr[0]),
                       col + 1 < cols ? __half2float(ptr[1]) : 0.0f);
}

__device__ __forceinline__ void store_gemm_half(half* C, int M, int N, int row, int col,
                                              float sum, float alpha, float beta) {
    if (row < M && col < N) {
        const size_t index = static_cast<size_t>(row) * N + col;
        const float previous = beta != 0.0f ? __half2float(C[index]) : 0.0f;
        C[index] = __float2half_rn(fmaf(alpha, sum, beta * previous));
    }
}

// Comparison to gemm_smem_reg_kernel: same 32x32 FP32 shared tiles and 2x2
// outputs/thread. Launch dim3(tile_side/2, tile_side/2) = (16,16).
// Pack each adjacent pair of global halfs into one 32-bit half2 load.
__global__ void gemm_smem_reg_half2_kernel(const half* A, const half* B, half* C,
                                         int M, int N, int K, float alpha, float beta) {
    const int local_row = threadIdx.y * 2;
    const int local_col = threadIdx.x * 2;
    const int tile_row = blockIdx.y * tile_side;
    const int tile_col = blockIdx.x * tile_side;
    const int row = tile_row + local_row;
    const int col = tile_col + local_col;
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    __shared__ __align__(8) float sA[tile_side][tile_side];
    __shared__ __align__(8) float sB[tile_side][tile_side];

    for (int k_base = 0; k_base < K; k_base += tile_side) {
        // Stage 1: four 32-bit input loads/thread replace eight scalar half
        // loads. Each converted float2 is written directly to shared memory.
        const float2 a_row0 = load_gemm_half2(A, M, K, row, k_base + local_col);
        const float2 a_row1 = load_gemm_half2(A, M, K, row + 1, k_base + local_col);
        const float2 b_row0 = load_gemm_half2(B, K, N, k_base + local_row, col);
        const float2 b_row1 = load_gemm_half2(B, K, N, k_base + local_row + 1, col);
        *reinterpret_cast<float2*>(&sA[local_row][local_col]) = a_row0;
        *reinterpret_cast<float2*>(&sA[local_row + 1][local_col]) = a_row1;
        *reinterpret_cast<float2*>(&sB[local_row][local_col]) = b_row0;
        *reinterpret_cast<float2*>(&sB[local_row + 1][local_col]) = b_row1;
        __syncthreads();

        // Stage 2: no rA[2]/rB[2] fill loop or 2x2 accumulator loop. B's two
        // adjacent FP32 columns form one float2 load. A's two rows are strided,
        // so they require two separate scalar shared loads with this layout.
        for (int k = 0; k < tile_side; ++k) {
            const float a0 = sA[local_row][k];
            const float a1 = sA[local_row + 1][k];
            const float2 b = *reinterpret_cast<const float2*>(&sB[k][local_col]);
            acc00 = fmaf(a0, b.x, acc00);
            acc01 = fmaf(a0, b.y, acc01);
            acc10 = fmaf(a1, b.x, acc10);
            acc11 = fmaf(a1, b.y, acc11);
        }
        __syncthreads();
    }

    // Stage 3: same scalar FP16 epilogue as the baseline, with independent guards.
    store_gemm_half(C, M, N, row, col, acc00, alpha, beta);
    store_gemm_half(C, M, N, row, col + 1, acc01, alpha, beta);
    store_gemm_half(C, M, N, row + 1, col, acc10, alpha, beta);
    store_gemm_half(C, M, N, row + 1, col + 1, acc11, alpha, beta);
}

// Alternative to the half2 path: widen global loads to 16 bytes and compute
// eight adjacent outputs/thread; keep FP32 shared tiles and scalar FMA math.
constexpr int F4_TILE_M = 32;
constexpr int F4_TILE_N = 32;
constexpr int F4_TILE_K = 32;
constexpr int F4_HALFS = sizeof(float4) / sizeof(half);
constexpr int F4_THREADS_X = F4_TILE_N / F4_HALFS;
constexpr int F4_THREADS_Y = F4_TILE_M;

// Copy eight FP16 bit patterns, NOT four numerically converted FP32 values.
// dst must be 16B aligned.
__device__ __forceinline__ void load_half8(const half* src, int rows, int cols,
                                         int row, int col, half* dst) {
    if (row < rows && col < cols) {
        const half* ptr = src + static_cast<size_t>(row) * cols + col;
        if (col + F4_HALFS <= cols &&
            (reinterpret_cast<std::uintptr_t>(ptr) & (alignof(float4) - 1)) == 0) {
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(ptr);
            return;
        }
    }
    // Misaligned rows and partial M/N/K tiles take a safe, zero-padded fallback.
    #pragma unroll
    for (int i = 0; i < F4_HALFS; ++i) {
        dst[i] = (row < rows && col + i < cols)
                     ? src[static_cast<size_t>(row) * cols + col + i]
                     : __float2half_rn(0.0f);
    }
}

// Global memory is FP16, but the shared tiles are FP32. First load one raw
// float4 (eight halfs) into aligned thread-local storage, then convert each half.
__device__ __forceinline__ void load_half8_f32(const half* src, int rows, int cols,
                                             int row, int col, float* dst) {
    __align__(16) half values[F4_HALFS];
    load_half8(src, rows, cols, row, col, values);
    #pragma unroll
    for (int i = 0; i < F4_HALFS; ++i) {
        dst[i] = __half2float(values[i]);
    }
}

// Launch ONLY with dim3(4, 32). One CTA computes C[32,32], stepping K by 32.
// A warp spans eight thread-y rows, with four thread-x lanes per row:
//   tx:          0       1       2       3
//   half cols:  0..7    8..15   16..23   24..31
// Four lanes load 4 * 16B = 64B (32 halfs), then write 32 floats to shared memory.
__global__ void gemm_f4_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                            float beta) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int local_col = tx * F4_HALFS;
    const int row = blockIdx.y * F4_TILE_M + ty;
    const int tile_col = blockIdx.x * F4_TILE_N;
    const int col = tile_col + local_col;
    float acc[F4_HALFS] = {0.0f};  // Thread owns C[row, col .. col+7].

    // Exactly two unpadded 32x32 FP32 shared tiles, as requested: 8192 bytes total.
    __shared__ float sA[F4_TILE_M][F4_TILE_K];
    __shared__ float sB[F4_TILE_K][F4_TILE_N];

    for (int k_base = 0; k_base < K; k_base += F4_TILE_K) {
        // Stage 1: each thread loads/converts eight A values and eight B values.
        // All 4x32 threads together fill each 32x32 shared tile exactly once.
        load_half8_f32(A, M, K, row, k_base + local_col, &sA[ty][local_col]);
        load_half8_f32(B, K, N, k_base + ty, col, &sB[ty][local_col]);
        __syncthreads();  // All rows must be ready before any thread consumes them.

        // Stage 2: thread (tx,ty) computes C[row, tile_col + 8 * tx + i], i=0..7.
        // Read FP32 shared values directly; reuse one A value for eight outputs.
        #pragma unroll 4
        for (int k = 0; k < F4_TILE_K; ++k) {
            const float a = sA[ty][k];
            #pragma unroll
            for (int i = 0; i < F4_HALFS; ++i) {
                acc[i] = fmaf(a, sB[k][local_col + i], acc[i]);
            }
        }
        __syncthreads();  // No overwrite until every warp finishes this K tile.
    }

    // Stage 3: apply alpha/beta once. Edge threads participated in all barriers.
    if (row < M && col < N) {
        __align__(16) half output[F4_HALFS];
        if (beta != 0.0f) {
            load_half8(C, M, N, row, col, output);
        }
        #pragma unroll
        for (int i = 0; i < F4_HALFS; ++i) {
            const float previous = beta != 0.0f ? __half2float(output[i]) : 0.0f;
            output[i] = __float2half_rn(fmaf(alpha, acc[i], beta * previous));
        }
        half* ptr = C + static_cast<size_t>(row) * N + col;
        if (col + F4_HALFS <= N &&
            (reinterpret_cast<std::uintptr_t>(ptr) & (alignof(float4) - 1)) == 0) {
            *reinterpret_cast<float4*>(ptr) = *reinterpret_cast<const float4*>(output);
        } else {
            #pragma unroll
            for (int i = 0; i < F4_HALFS; ++i) {
                if (col + i < N) {
                    ptr[i] = output[i];
                }
            }
        }
    }
}

// 3. Warp-level Tensor Cores: WMMA first, then asynchronous/swizzled staging.
__global__ void gemm_wmma_smem_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                      float alpha, float beta) {
    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * TILE;
    const int tile_col = blockIdx.x * TILE;

    // The output remains shared: acc_frag is float, C is half, and the scalar
    // epilogue needs known row/column ownership.
    __shared__ __align__(32) float sProduct[TILE][TILE];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_base = 0; k_base < K; k_base += TILE) {
        __shared__ __align__(32) half sA[TILE][TILE];
        __shared__ __align__(32) half sB[TILE][TILE];
        for (int element = lane; element < TILE * TILE; element += WARP_SIZE) {
            const int r = element / TILE;
            const int c = element % TILE;
            sA[r][c] = (tile_row + r < M && k_base + c < K) ? A[(tile_row + r) * K + k_base + c]
                                                            : __float2half_rn(0.0f);
            sB[r][c] = (k_base + r < K && tile_col + c < N) ? B[(k_base + r) * N + tile_col + c]
                                                            : __float2half_rn(0.0f);
        }
        __syncwarp(FULL_WARP_MASK);
        wmma::load_matrix_sync(a_frag, &sA[0][0], TILE);
        wmma::load_matrix_sync(b_frag, &sB[0][0], TILE);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp(FULL_WARP_MASK);
    }

    wmma::store_matrix_sync(&sProduct[0][0], acc_frag, TILE, wmma::mem_row_major);
    __syncwarp(FULL_WARP_MASK);

    // Each lane converts and stores its share of the FP32 output tile.
    for (int element = lane; element < TILE * TILE; element += WARP_SIZE) {
        const int r = element / TILE;
        const int c = element % TILE;
        const int row = tile_row + r;
        const int col = tile_col + c;
        if (row < M && col < N) {
            const int output_index = row * N + col;
            const float previous = beta != 0.0f ? __half2float(C[output_index]) : 0.0f;
            C[output_index] = __float2half_rn(alpha * sProduct[r][c] + beta * previous);
        }
    }
}

// Direct gmem to wmma loads: no shared memory, no staging, no padding. Each warp computes one 16x16 tile.
__global__ void gemm_wmma_gmem_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                      float alpha, float beta) {
    int lane = threadIdx.x % WARP_SIZE;
    int row_tile = blockIdx.y * TILE;
    int col_tile = blockIdx.x * TILE;
    __shared__ __align__(32) float sP[TILE][TILE];
    // init fragments
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // load to fragments
    for (int k_base = 0; k_base < K; k_base += TILE) {
        wmma::load_matrix_sync(a_frag, A + row_tile * K + k_base, K);
        wmma::load_matrix_sync(b_frag, B + k_base * N + col_tile, N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    // store back
    wmma::store_matrix_sync(&sP[0][0], acc, TILE, wmma::mem_row_major);
    __syncwarp(0xffffffffu);

    for (int element = lane; element < TILE * TILE; element += WARP_SIZE) {
        int row_id = element / TILE;
        int col_id = element % TILE;
        int global_row_id = row_tile + row_id;
        int global_col_id = col_tile + col_id;
        if (global_row_id < M && global_col_id < N) {
            float C_prev =
                (beta == 0.0f) ? 0.0f : beta * __half2float(C[global_col_id + global_row_id * N]);
            C[global_col_id + global_row_id * N] =
                __float2half_rn(fmaf(alpha, sP[row_id][col_id], C_prev));
        }
    }
}

// Compared with the 16x16 WMMA paths, reuse operands across four output tiles.
constexpr int WMMA_REG_TILE = 2 * TILE;

// Still one warp/block, but a 2x2 grid of 16x16 accumulators covers a 32x32
// output tile. Each A fragment serves two column tiles; each B fragment serves
// two row tiles.
__global__ void gemm_wmma_smem_reg_kernel(const half* A, const half* B, half* C,
                                        int M, int N, int K, float alpha, float beta) {
    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * WMMA_REG_TILE;
    const int tile_col = blockIdx.x * WMMA_REG_TILE;
    __shared__ __align__(32) half sA[2][TILE][TILE];
    __shared__ __align__(32) half sB[2][TILE][TILE];
    __shared__ __align__(32) float sProduct[TILE][TILE];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k_base = 0; k_base < K; k_base += TILE) {
        // Stage 1: cooperatively stage two A row tiles and two B column tiles.
        // Zero-padding retains the baseline's arbitrary-shape/pointer support.
        for (int element = lane; element < TILE * TILE; element += WARP_SIZE) {
            const int r = element / TILE;
            const int c = element % TILE;
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int row = tile_row + t * TILE + r;
                const int col = tile_col + t * TILE + c;
                sA[t][r][c] = (row < M && k_base + c < K)
                    ? A[row * K + k_base + c] : __float2half_rn(0.0f);
                sB[t][r][c] = (k_base + r < K && col < N)
                    ? B[(k_base + r) * N + col] : __float2half_rn(0.0f);
            }
        }
        __syncwarp(FULL_WARP_MASK);

        // Stage 2: four operand loads feed four MMAs instead of eight loads.
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            wmma::load_matrix_sync(a_frag[t], &sA[t][0][0], TILE);
            wmma::load_matrix_sync(b_frag[t], &sB[t][0][0], TILE);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
        __syncwarp(FULL_WARP_MASK);
    }

    // Stage 3: reuse the baseline's single FP32 scratch tile for all four
    // accumulators. This avoids reserving four output tiles in shared memory.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&sProduct[0][0], acc[i][j], TILE,
                                    wmma::mem_row_major);
            __syncwarp(FULL_WARP_MASK);
            for (int element = lane; element < TILE * TILE; element += WARP_SIZE) {
                const int r = element / TILE;
                const int c = element % TILE;
                const int row = tile_row + i * TILE + r;
                const int col = tile_col + j * TILE + c;
                if (row < M && col < N) {
                    const int index = row * N + col;
                    const float previous = beta != 0.0f ? __half2float(C[index]) : 0.0f;
                    C[index] = __float2half_rn(alpha * sProduct[r][c] + beta * previous);
                }
            }
            // Finish all readers before the next accumulator overwrites it.
            __syncwarp(FULL_WARP_MASK);
        }
    }
}

// Compared with the one-warp register-tile path, use four warps and overlap
// global-to-shared copies with Tensor Core work through several input buffers.
// Four warps cover a 64x64 output tile; each warp owns a 32x32 quadrant.
// Copy 32 K values/stage, with a ring of asynchronous global->shared buffers.
constexpr int PIPE_M = 64;
constexpr int PIPE_N = 64;
constexpr int PIPE_STAGES = 3;  // Fastest of the tested 2/3/4-stage variants on SM120.
constexpr int PIPE_A_STRIDE = PIPE_K + 8;  // 40 halfs: row-start bank step 20.
constexpr int PIPE_B_STRIDE = PIPE_N + 8;  // 72 halfs: row-start bank step 4.
constexpr int PIPE_C_STRIDE = PIPE_N + 4;  // 68 floats: pad accumulator stores.

// Fast-path contract: M/N multiples of 64, positive K multiple of 32, and
// 16-byte-aligned A/B/C. Shared WMMA pointers remain 32-byte aligned; their
// padded leading dimensions are multiples of 8 halfs, as WMMA requires.
template <int Stages>
__global__ void gemm_wmma_pipeline_kernel(const half* A, const half* B, half* C,
                                        int M, int N, int K, float alpha, float beta) {
    static_assert(Stages >= 2 && Stages <= 4, "Use two to four pipeline stages");
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int tile_row = blockIdx.y * PIPE_M;
    const int tile_col = blockIdx.x * PIPE_N;
    const int warp_row = (warp / 2) * WMMA_REG_TILE;
    const int warp_col = (warp % 2) * WMMA_REG_TILE;
    const int k_tiles = K / PIPE_K;

    // The input ring is dead before the epilogue starts. Alias its storage
    // with the FP32 output tile rather than reserving both simultaneously.
    union SharedStorage {
        struct {
            half a[Stages][PIPE_M][PIPE_A_STRIDE];
            half b[Stages][PIPE_K][PIPE_B_STRIDE];
        } input;
        float product[PIPE_M][PIPE_C_STRIDE];
    };
    __shared__ __align__(32) SharedStorage shared;

    // Each lane copies four 16-byte vectors/stage: two from A, two from B.
    auto copy_stage = [&](int slot, int k_tile) {
        const bool valid = k_tile < k_tiles;
        const int k_base = k_tile * PIPE_K;
        #pragma unroll
        for (int vector = tid; vector < PIPE_M * PIPE_K / 8; vector += PIPE_THREADS) {
            const int row = vector / (PIPE_K / 8);
            const int col = (vector % (PIPE_K / 8)) * 8;
            const half* src = valid ? A + (tile_row + row) * K + k_base + col : A;
            gemm_copy_async_16(&shared.input.a[slot][row][col], src, valid);
        }
        #pragma unroll
        for (int vector = tid; vector < PIPE_K * PIPE_N / 8; vector += PIPE_THREADS) {
            const int row = vector / (PIPE_N / 8);
            const int col = (vector % (PIPE_N / 8)) * 8;
            const half* src = valid ? B + (k_base + row) * N + tile_col + col : B;
            gemm_copy_async_16(&shared.input.b[slot][row][col], src, valid);
        }
        gemm_copy_async_commit();
    };

    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag[2][2];
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Stage 1: prime Stages-1 buffers. Tail prefetches zero-fill safely.
    #pragma unroll
    for (int stage = 0; stage < Stages - 1; ++stage) {
        copy_stage(stage, stage);
    }

    // Stage 2: consume the oldest completed buffer while fetching a future
    // tile into a different slot. The end barrier protects buffer recycling.
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        gemm_copy_async_wait<Stages - 2>();
        __syncthreads();  // Publish completed copies from all four warps.
        const int slot = k_tile % Stages;
        copy_stage((k_tile + Stages - 1) % Stages, k_tile + Stages - 1);

        // Prefetch both K=16 register groups before issuing their MMAs. The
        // second group's shared loads can complete while the first computes.
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                wmma::load_matrix_sync(a_frag[group][t],
                    &shared.input.a[slot][warp_row + t * TILE][group * TILE], PIPE_A_STRIDE);
                wmma::load_matrix_sync(b_frag[group][t],
                    &shared.input.b[slot][group * TILE][warp_col + t * TILE], PIPE_B_STRIDE);
            }
        }
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], a_frag[group][i], b_frag[group][j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Stage 3: drain ALL tail copies before aliasing the input buffers with
    // output storage. Each warp materializes only its own 32x32 quadrant.
    gemm_copy_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&shared.product[warp_row + i * TILE][warp_col + j * TILE],
                                    acc[i][j], PIPE_C_STRIDE, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Stage 4: vectorized epilogue. Each lane owns eight adjacent half outputs
    // per iteration. float4 transfers raw FP16 bits; arithmetic stays FP32.
    for (int vector = tid; vector < PIPE_M * PIPE_N / 8; vector += PIPE_THREADS) {
        const int row = vector / (PIPE_N / 8);
        const int col = (vector % (PIPE_N / 8)) * 8;
        const int index = (tile_row + row) * N + tile_col + col;
        __align__(16) half previous[8];
        __align__(16) half output[8];
        if (beta != 0.0f) {
            *reinterpret_cast<float4*>(previous) = *reinterpret_cast<const float4*>(C + index);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float value = beta != 0.0f ? __half2float(previous[i]) : 0.0f;
            output[i] = __float2half_rn(alpha * shared.product[row][col + i] + beta * value);
        }
        *reinterpret_cast<float4*>(C + index) = *reinterpret_cast<const float4*>(output);
    }
}

// Alternative to the padded WMMA pipeline: XOR-swizzle shared addresses and
// explicitly load register fragments with ldmatrix before mma.sync.
constexpr int MMA_STAGES = 3;

// Same eligible shapes/alignment as the WMMA pipeline: 64x64 output, BK=32,
// four warps, and 16-byte-aligned global pointers. Each warp owns 32x32 outputs.
template <int Stages>
__global__ void gemm_wmma_swizzle_kernel(const half* A, const half* B, half* C,
                                      int M, int N, int K, float alpha, float beta) {
    static_assert(Stages >= 2 && Stages <= 6, "Use two to six async-copy stages");
    union SharedStorage {
        struct {
            half a[Stages][PIPE_M][PIPE_K];
            half b[Stages][PIPE_K][PIPE_N];
        } input;
        float product[PIPE_M][PIPE_C_STRIDE];
    };
    __shared__ __align__(32) SharedStorage shared;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int warp_row = (warp / 2) * WMMA_REG_TILE, warp_col = (warp % 2) * WMMA_REG_TILE;
    const int tile_row = blockIdx.y * PIPE_M, tile_col = blockIdx.x * PIPE_N;
    const int k_tiles = K / PIPE_K;

    auto copy_stage = [&](int slot, int k_tile) {
        const bool valid = k_tile < k_tiles;
        #pragma unroll
        for (int v = tid; v < PIPE_M * PIPE_K / 8; v += PIPE_THREADS) {
            const int row = v / (PIPE_K / 8), col = (v % (PIPE_K / 8)) * 8;
            // Permute WHOLE eight-half vectors, never elements within a vector.
            // A has four vectors/row: row pairs select different vector columns.
            const int physical_col = col ^ (((row >> 1) & 3) * 8);
            const half* src = valid ? A + (tile_row + row) * K + k_tile * PIPE_K + col : A;
            gemm_copy_async_16(&shared.input.a[slot][row][physical_col], src, valid);
        }
        #pragma unroll
        for (int v = tid; v < PIPE_K * PIPE_N / 8; v += PIPE_THREADS) {
            const int row = v / (PIPE_N / 8), col = (v % (PIPE_N / 8)) * 8;
            // B has eight vectors/row: each of eight rows selects a different
            // 16-byte bank group. Both copy and ldmatrix must use the SAME XOR.
            const int physical_col = col ^ ((row & 7) * 8);
            const half* src = valid ? B + (k_tile * PIPE_K + row) * N + tile_col + col : B;
            gemm_copy_async_16(&shared.input.b[slot][row][physical_col], src, valid);
        }
        gemm_copy_async_commit();
    };

    unsigned a_frag[2][2][4], b_frag[2][2][4];  // [register buffer][16x16 tile][packed register]
    float acc[2][4][4] = {};                    // two M tiles x four N=8 tiles x four outputs/lane
    auto load_fragments = [&](int buffer, int slot, int group) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = warp_row + i * TILE + lane % TILE;
            const int col = group * TILE + (lane / TILE) * 8;
            gemm_mma_load_a(a_frag[buffer][i],
                &shared.input.a[slot][row][col ^ (((row >> 1) & 3) * 8)]);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = group * TILE + lane % TILE;
            const int col = warp_col + j * TILE + (lane / TILE) * 8;
            gemm_mma_load_b(b_frag[buffer][j],
                &shared.input.b[slot][row][col ^ ((row & 7) * 8)]);
        }
    };
    auto compute_group = [&](int buffer) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                gemm_mma_16x8(acc[i][j], a_frag[buffer][i], &b_frag[buffer][j / 2][(j % 2) * 2]);
            }
        }
    };

    #pragma unroll
    for (int slot = 0; slot < Stages - 1; ++slot) copy_stage(slot, slot);
    gemm_copy_async_wait<Stages - 2>();
    __syncthreads();
    load_fragments(0, 0, 0);
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        copy_stage((k_tile + Stages - 1) % Stages, k_tile + Stages - 1);
        load_fragments(1, k_tile % Stages, 1);  // Load the next K=16 group before the current MMA.
        compute_group(0);

        // All current-stage shared reads are now issued. Wait/publish the next
        // stage and protect buffer reuse, then prefetch its first register group
        // BEFORE computing the old stage's last group. That last MMA uses only
        // buffer 1 registers, hiding some latency of the buffer 0 shared loads.
        gemm_copy_async_wait<Stages - 2>();
        __syncthreads();
        load_fragments(0, (k_tile + 1) % Stages, 0);
        compute_group(1);
    }
    // Tail zero-fills still write shared memory. Drain every copy from every
    // lane before aliasing the input ring with the epilogue's output scratch.
    gemm_copy_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
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
    for (int v = tid; v < PIPE_M * PIPE_N / 8; v += PIPE_THREADS) {
        const int row = v / (PIPE_N / 8), col = (v % (PIPE_N / 8)) * 8;
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

}  // namespace
