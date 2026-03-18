# asterisk enforces keyword-only arguments
def count_flops(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    # $$N_{\text{FLOPs}} = BK(H(6D+1) + 24D) + B(K + 20)$$
    return batch_size * n_layers * (hidden * (6 * dim + 1) + 24 * dim) + batch_size * (n_layers + 20)


