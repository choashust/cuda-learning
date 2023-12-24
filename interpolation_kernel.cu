#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(  // device修饰 gloabal修饰
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats, //(N, 8 , F) 
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points, //(N, 3)
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp //(N, F)
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n>=feats.size(0) || f>=feats.size(2))
        return;
    
    // 三线性插值 并行运算
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
}

template <typename scalar_t>
__global__ void trilinear_bw_kernel(  // device修饰 gloabal修饰
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp, //(N, F)
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats, //(N, 8 , F) 
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points, //(N, 3)
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats //(N, 8, F)
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n>=feats.size(0) || f>=feats.size(2))
        return;
    
    // 三线性插值 并行运算
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;


    dL_dfeats[n][0][f] = (1-u)*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1-u)*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1-u)*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1-u)*d*dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u*d*dL_dfeat_interp[n][f];
}

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
){  //(N, 8 , F) (N, 3)
    const int N = feats.size(0), F = feats.size(2); 

    //python feat_interp = torch.zeros(N, F, dtype=torch.float32, device='cuda')
    torch::Tensor feat_interp = torch::empty({N, F}, feats.options());  
    //指定type\device torch::dtype(torch::kInt32).device(feats.device)

    const dim3 threads(16, 16); // thread 最多定义256个  N个并行 F个并行
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y); //最少有1个block

    // kernel调用 数据类型 函数名称 , FLOATING_TYPES 可以32或者64位float
    // scalar_t指代32或者64位float | 2,3 表示维度 | torch::RestrictPtrTraits 元素没有交集 | size_t 会根据scalar_t填充
    // kernel函数没有返回值 需要输入输出都传入 
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return feat_interp;
}

torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){  //(N, 8 , F) (N, 3)
    const int N = feats.size(0), F = feats.size(2); 

    //python feat_interp = torch.zeros(N, F, dtype=torch.float32, device='cuda')
    torch::Tensor dL_dfeats = torch::zeros({N, 8, F}, feats.options());  
    //指定type\device torch::dtype(torch::kInt32).device(feats.device)

    const dim3 threads(16, 16); // thread 最多定义256个  N个并行 F个并行
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y); //最少有1个block

    // kernel调用 数据类型 函数名称 , FLOATING_TYPES 可以32或者64位float
    // scalar_t指代32或者64位float | 2,3 表示维度 | torch::RestrictPtrTraits 元素没有交集 | size_t 会根据scalar_t填充
    // kernel函数没有返回值 需要输入输出都传入 
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu", 
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dfeats;
}