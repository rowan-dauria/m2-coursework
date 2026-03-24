# asterisk enforces keyword-only arguments
def count_flops(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    # $$N_{\text{FLOPs}} = BK(H(6D+1) + 24D) + B(K + 20)$$
    # Counts exp, neg, sub, mul on all D dims (full vectors as output by MLP)
    return batch_size * n_layers * (hidden * (6 * dim + 1) + 24 * dim) + batch_size * (n_layers + 20)


def count_flops_alt(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    # $$N_{\text{FLOPs}} = BK(H(3D+1) + 12D) + B(K + 20)$$
    # MLP sees D/2 inputs, outputs D (= 2 * D/2 for s and t); transform on D/2 dims only
    return batch_size * n_layers * (hidden * (3 * dim + 1) + 12 * dim) + batch_size * (n_layers + 20)


