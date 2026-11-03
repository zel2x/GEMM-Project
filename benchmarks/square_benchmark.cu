// Direct, reproducible 1024^3 comparisons; implementation headers are unchanged.
#include "../src/lib/gemm_progressive_kernel.cuh"
#include "../src/lib/gemm_cublas.cuh"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(call) do { const auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)

namespace {
constexpr int Size=1024, Guard=64, Count=Size*Size, Variants=10;
constexpr size_t EvictionBytes=256ull*1024*1024;
constexpr float Canary=123.0f;
struct Variant { const char* kernel; dim3 grid; dim3 block; };
const Variant variants[Variants]={
    {"gemm_base_kernel",{32,128,1},{32,8,1}},
    {"gemm_smem_kernel",{32,32,1},{32,32,1}},
    {"gemm_smem_reg_kernel",{32,32,1},{16,16,1}},
    {"gemm_smem_reg_half2_kernel",{32,32,1},{16,16,1}},
    {"gemm_wmma_smem_kernel",{64,64,1},{32,1,1}},
    {"gemm_wmma_gmem_kernel",{64,64,1},{32,1,1}},
    {"gemm_wmma_pipeline_kernel<3>",{16,16,1},{128,1,1}},
    {"gemm_wmma_swizzle_kernel<3>",{16,16,1},{128,1,1}},
    {"cuBLAS (gemm_cublas)",{0,0,0},{0,0,0}},
    {"gemm_mma_llm_kernel<64, 128, 3, 32>",{128,1,1},{128,1,1}},
};

void launch(int v,const half* a,const half* b,half* c) {
    const dim3 grid=variants[v].grid,block=variants[v].block;
    switch(v) {
    case 0: gemm_base_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 1: gemm_smem_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 2: gemm_smem_reg_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 3: gemm_smem_reg_half2_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 4: gemm_wmma_smem_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 5: gemm_wmma_gmem_kernel<<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 6: gemm_wmma_pipeline_kernel<3><<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 7: gemm_wmma_swizzle_kernel<3><<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 8: gemm_cublas(a,b,c,Size,Size,Size,1.f,0.f);break;
    case 9: gemm_mma_llm_kernel<64,128,3,32><<<grid,block>>>(a,b,c,Size,Size,Size,1.f,0.f);break;
    default: throw std::runtime_error("Unknown variant");
    }
    CK(cudaGetLastError());
}

struct Data {
    std::vector<half> ha,hb,hc;
    half *aa=nullptr,*bb=nullptr,*cc=nullptr;
    Data():ha(Count+2*Guard,__float2half(Canary)),hb(ha),hc(ha) {
        uint32_t seed=1729+3*Size;
        for(auto* values:{&ha,&hb,&hc})
            for(int i=0;i<Count;++i) {
                seed=1664525u*seed+1013904223u;
                (*values)[Guard+i]=__float2half_rn(float(seed>>8)*0x1p-24f-.5f);
            }
        const size_t bytes=ha.size()*sizeof(half);
        CK(cudaMalloc(&aa,bytes));CK(cudaMalloc(&bb,bytes));CK(cudaMalloc(&cc,bytes));
        CK(cudaMemcpy(aa,ha.data(),bytes,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bb,hb.data(),bytes,cudaMemcpyHostToDevice));reset();
    }
    half* a()const{return aa+Guard;} half* b()const{return bb+Guard;} half* c()const{return cc+Guard;}
    void reset(){CK(cudaMemcpy(cc,hc.data(),hc.size()*sizeof(half),cudaMemcpyHostToDevice));}
    ~Data(){cudaFree(aa);cudaFree(bb);cudaFree(cc);}
};

std::vector<float> reference(const Data& d) {
    std::vector<float> a(Count),b(Count),c(Count);
    for(int i=0;i<Count;++i){a[i]=__half2float(d.ha[Guard+i]);b[i]=__half2float(d.hb[Guard+i]);}
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    CK(cudaMalloc(&da,Count*sizeof(float)));CK(cudaMalloc(&db,Count*sizeof(float)));CK(cudaMalloc(&dc,Count*sizeof(float)));
    CK(cudaMemcpy(da,a.data(),Count*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),Count*sizeof(float),cudaMemcpyHostToDevice));
    cublasHandle_t h;check_cublas(cublasCreate(&h));check_cublas(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
    const float alpha=1.f,beta=0.f;
    check_cublas(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Size,Size,Size,&alpha,
        db,CUDA_R_32F,Size,da,CUDA_R_32F,Size,&beta,dc,CUDA_R_32F,Size,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());CK(cudaMemcpy(c.data(),dc,Count*sizeof(float),cudaMemcpyDeviceToHost));
    check_cublas(cublasDestroy(h));cudaFree(da);cudaFree(db);cudaFree(dc);
    // Independently audit 64 reference outputs using CPU FP64 dot products.
    for(int spot=0;spot<64;++spot) {
        const int row=(spot*137+17)%Size,col=(spot*251+29)%Size;
        double sum=0.;for(int k=0;k<Size;++k)sum+=double(a[row*Size+k])*double(b[k*Size+col]);
        if(std::abs(double(c[row*Size+col])-sum)>1.e-5+1.e-5*std::abs(sum))
            throw std::runtime_error("FP32 reference failed the FP64 spot audit");
    }
    return c;
}

bool guards(const std::vector<half>& values) {
    for(int i=0;i<Guard;++i)
        if(__half2float(values[i])!=Canary||__half2float(values[Guard+Count+i])!=Canary)return false;
    return true;
}

void validate(const std::string& path) {
    Data d;const auto ref=reference(d);std::ofstream out(path);if(!out)throw std::runtime_error("Cannot write correctness results");
    out<<std::setprecision(12)<<"{\"rtol\":0.002,\"atol\":0.002,\"nrms_limit\":0.001,\"fp64_reference_spots\":64,\"cases\":[\n";
    bool all=true;
    for(int v=0;v<Variants;++v) {
        d.reset();launch(v,d.a(),d.b(),d.c());CK(cudaDeviceSynchronize());
        std::vector<half> got(d.hc.size()),a(d.ha.size()),b(d.hb.size());
        CK(cudaMemcpy(got.data(),d.cc,got.size()*sizeof(half),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(a.data(),d.aa,a.size()*sizeof(half),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),d.bb,b.size()*sizeof(half),cudaMemcpyDeviceToHost));
        double squared=0.,ref_squared=0.,maximum=0.;size_t violations=0;
        for(int i=0;i<Count;++i) {
            const float value=__half2float(got[Guard+i]);const double error=std::abs(double(value)-ref[i]);
            if(!std::isfinite(value)||error>.002+.002*std::abs(double(ref[i])))++violations;
            maximum=std::max(maximum,error);squared+=error*error;ref_squared+=double(ref[i])*ref[i];
        }
        const double nrms=std::sqrt(squared/ref_squared);
        const bool canaries=guards(got)&&guards(a)&&guards(b);
        const bool inputs=std::memcmp(a.data(),d.ha.data(),a.size()*sizeof(half))==0&&
                          std::memcmp(b.data(),d.hb.data(),b.size()*sizeof(half))==0;
        const bool passed=violations==0&&nrms<=.001&&canaries&&inputs;all&=passed;
        out<<(v?",\n":"")<<"{\"id\":"<<v<<",\"kernel\":\""<<variants[v].kernel
           <<"\",\"pass\":"<<(passed?"true":"false")<<",\"max_abs_error\":"<<maximum
           <<",\"nrms\":"<<nrms<<",\"tolerance_violations\":"<<violations
           <<",\"canaries_intact\":"<<(canaries?"true":"false")<<",\"inputs_unchanged\":"<<(inputs?"true":"false")<<"}";
        std::cout<<"CORRECTNESS "<<v<<" "<<variants[v].kernel<<" "<<(passed?"PASS":"FAIL")<<" nrms="<<nrms<<std::endl;
    }
    out<<"\n],\"passed\":"<<(all?"true":"false")<<"}\n";out.close();
    if(!all)throw std::runtime_error("Correctness failed; profiling is blocked");
}

__global__ void initialize_eviction(uint32_t* data,size_t count) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        uint32_t x=uint32_t(i)*747796405u+2891336453u;
        x=((x>>((x>>28u)+4u))^x)*277803737u;data[i]=(x>>22u)^x;
    }
}

__global__ void sweep_eviction(const uint32_t* data,uint32_t* sink,uint32_t lines) {
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid%32,warp=tid/32;
    const unsigned warps=gridDim.x*blockDim.x/32;uint32_t sum=0;
    for(uint32_t logical=warp;logical<lines;logical+=warps) {
        // Odd-multiplier permutation covers every 128-byte line exactly once.
        const uint32_t physical=(logical*1664525u+1013904223u)&(lines-1);
        uint32_t value;const uint32_t* address=data+size_t(physical)*32+lane;
        asm volatile("ld.global.ca.u32 %0, [%1];":"=r"(value):"l"(address):"memory");
        sum^=value;
    }
    sink[tid]=sum;
}

void profile() {
    Data d;cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
    const int blocks=prop.multiProcessorCount*16,threads=256;
    uint32_t *buffer=nullptr,*sink=nullptr;
    CK(cudaMalloc(&buffer,EvictionBytes));CK(cudaMalloc(&sink,size_t(blocks)*threads*sizeof(uint32_t)));
    initialize_eviction<<<blocks,threads>>>(buffer,EvictionBytes/sizeof(uint32_t));CK(cudaDeviceSynchronize());
    for(int v=0;v<Variants;++v) {
        d.reset();for(int w=0;w<10;++w)launch(v,d.a(),d.b(),d.c());CK(cudaDeviceSynchronize());
    }
    for(int trial=0;trial<9;++trial)
        for(int i=0;i<Variants;++i) {
            const int v=(i+trial)%Variants;
            d.reset();sweep_eviction<<<blocks,threads>>>(buffer,sink,EvictionBytes/128);
            CK(cudaGetLastError());CK(cudaDeviceSynchronize());
            const std::string label="square-v"+(v<10?std::string("0"):std::string())+std::to_string(v)+
                                    "-t"+(trial<10?std::string("0"):std::string())+std::to_string(trial);
            nvtxRangePushA("gemm-profile");nvtxRangePushA(label.c_str());
            launch(v,d.a(),d.b(),d.c());CK(cudaDeviceSynchronize());
            nvtxRangePop();nvtxRangePop();
        }
    cudaFree(buffer);cudaFree(sink);
    std::cout<<"PROFILE_COMPLETE variants=10 trials=9 logical_calls=90"<<std::endl;
}

void list() {
    std::cout<<"[";
    for(int v=0;v<Variants;++v) {
        const auto& x=variants[v];
        std::cout<<(v?",":"")<<"{\"id\":"<<v<<",\"kernel\":\""<<x.kernel<<"\",\"grid\":["<<x.grid.x<<","<<x.grid.y<<","<<x.grid.z
            <<"],\"block\":["<<x.block.x<<","<<x.block.y<<","<<x.block.z<<"]}";
    }
    std::cout<<"]\n";
}
}  // namespace

int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--list"){list();return 0;}
        if(argc==3&&std::string(argv[1])=="--validate"){validate(argv[2]);return 0;}
        if(argc==2&&std::string(argv[1])=="--profile"){profile();return 0;}
        std::cerr<<"Usage: square_benchmark --list | --validate output.json | --profile\n";return 2;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
