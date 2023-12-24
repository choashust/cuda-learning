import torch 
import cppcuda_tutorial
import time

def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp

# 包装forward和backward函数
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):  # ctx 保存反向传播的参数
        feat_interp = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)

        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp): # 反向传播 ctx + forward的返回值梯度
        feats, points = ctx.saved_tensors

        dL_dfeats = cppcuda_tutorial.trilinear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)

        return dL_dfeats, None  # None表示不会对forward的入参points做调整, 和 foward的入参数量保持一致
    
if __name__ == "__main__":
    N = 65536; F = 256
    rand = torch.rand(N, 8, F, device='cuda')
    
    # feats1 torch写法 feats2 cuda写法
    feats1 = rand.clone().requires_grad_()
    feats2 = rand.clone().requires_grad_()
    points = torch.rand(N, 3, device='cuda')*2-1

    t = time.time()
    out_cuda = Trilinear_interpolation_cuda.apply(feats2, points)
    torch.cuda.synchronize()
    print('cuda fw time', time.time()-t, 's')

    t = time.time()
    out_py = trilinear_interpolation_py(feats1, points)
    torch.cuda.synchronize()
    print('pytorch fw time', time.time()-t, 's')

    print('fw all close', torch.allclose(out_py, out_cuda))

    t = time.time()
    loss2 = out_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print('cuda bw time', time.time()-t, 's')

    t = time.time()
    loss1 = out_py.sum()
    loss1.backward()
    torch.cuda.synchronize()
    print('pytorch bw time', time.time()-t, 's')

    print('bw all close', torch.allclose(feats1.grad, feats2.grad))