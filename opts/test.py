import torch

def measure_memory(Q, subspace_hessian, flat_grad, method='original'):
    """
    测量执行特定矩阵运算方法时的GPU内存使用情况。

    Args:
        Q (torch.Tensor): 形状为 [N x K] 的张量。
        subspace_hessian (torch.Tensor): 形状为 [K x K] 的张量。
        flat_grad (torch.Tensor): 形状为 [N] 的张量。
        method (str): 使用的方法，'original' 或 'modified'。

    Returns:
        torch.Tensor: 计算得到的向量 d。
    """
    # 清空未使用的缓存内存
    torch.cuda.empty_cache()
    
    # 重置峰值内存统计
    torch.cuda.reset_peak_memory_stats()
    
    # 获取操作前的内存使用情况
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    
    print(f"=== Method: {method} ===")
    #print(f"Memory allocated before operation: {before_allocated / (1024**3):.4f} GiB")
    #print(f"Memory reserved before operation: {before_reserved / (1024**3):.4f} GiB")
    
    # 执行矩阵运算
    with torch.no_grad():
        if method == 'original':
            # 原始方法：可能导致内存溢出
            d = - Q @ torch.inverse(subspace_hessian) @ (Q.T @ flat_grad)
        elif method == 'original2':
            # 原始方法：可能导致内存溢出
            d = - Q @ (torch.inverse(subspace_hessian) @ (Q.T @ flat_grad))
        elif method == 'modified':
            # 修改后的方法：减少内存使用
            QT_flat_grad = Q.T @ flat_grad                 # [K x N] @ [N] = [K]
            inv_subspace_hessian_QT_flat_grad = torch.linalg.solve(subspace_hessian, QT_flat_grad)  # 2 x 1
            d = - Q @ inv_subspace_hessian_QT_flat_grad  # 400 x 1
        else:
            raise ValueError("Method must be 'original' or 'modified'")
    
    # 获取操作后的内存使用情况
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    peak_memory = torch.cuda.max_memory_allocated()
    
    #print(f"Memory allocated after operation: {after_allocated / (1024**3):.4f} GiB")
    #print(f"Memory reserved after operation: {after_reserved / (1024**3):.4f} GiB")
    print(f"Peak memory usage during operation: {peak_memory / (1024**3):.4f} GiB\n")
    
    return d

# 示例用法
if __name__ == "__main__":
    # 假设 Q 是 [322401 x 2]，subspace_hessian 是 [2 x 2]，flat_grad 是 [322401]
    # set random seed for reproducibility
    torch.manual_seed(0)
    width = 322401
    width = 10000
    Q = torch.randn(width, 2, device='cuda')
    subspace_hessian = torch.randn(2, 2, device='cuda')
    flat_grad = torch.randn(width, device='cuda')
    
    # 测试原始方法（可能导致内存溢出）
    try:
        d_original = measure_memory(Q, subspace_hessian, flat_grad, method='original')
    except torch.cuda.OutOfMemoryError as e:
        print("Original method caused CUDA OutOfMemoryError.")
        torch.cuda.empty_cache()
    d_original2 = measure_memory(Q, subspace_hessian, flat_grad, method='original2')
    
    # 测试修改后的方法
    d_modified = measure_memory(Q, subspace_hessian, flat_grad, method='modified')
    # print(d_modified)  # 根据需要打印结果
    # check if the results are the same
    print(torch.norm(d_original - d_modified))
    print(torch.norm(d_original2 - d_original))
    print(torch.norm(d_original2 - d_modified))