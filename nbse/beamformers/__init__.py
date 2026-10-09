"""波束形成器：传统方法（解析解，无需训练）与 DOA 估计。

统一接口约定：
    enhance(multi_ch, doa) -> mono
        multi_ch : np.ndarray, shape [n_mics, T]
        doa      : float, 方位角（度）
        返回      : np.ndarray, shape [T]
"""
