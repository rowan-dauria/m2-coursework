# asterisk enforces keyword-only arguments
def count_flops(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    # $$N_{\text{FLOPs}} = BK(H(6D+1) + 24D) + B(K + 20)$$
    # Counts exp, neg, sub, mul on all D dims (full vectors as output by MLP)
    return batch_size * n_layers * (hidden * (6 * dim + 1) + 24 * dim) + batch_size * (n_layers + 20)


def count_flops_alt(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    # $$N_{\text{FLOPs}} = BK(H(3D+1) + 12D) + B(K + 20)$$
    # MLP sees D/2 inputs, outputs D (= 2 * D/2 for s and t); transform on D/2 dims only
    return batch_size * n_layers * (hidden * (3 * dim + 1) + 12 * dim) + batch_size * (n_layers + 20)


def print_flop_table(configs: list[dict], count_fn=count_flops) -> None:
    """Print a formatted table of FLOP counts for a list of configurations.

    Each entry in *configs* must be a dict with keys ``dim``, ``n_layers``,
    ``hidden``, and ``batch_size``.
    """
    print(f"\n{'dim':>4}  {'K':>3}  {'H':>4}  {'B':>5}  {'FLOPs':>12}")
    print("-" * 36)
    for cfg in configs:
        flops = count_fn(**cfg)
        print(f"{cfg['dim']:>4}  {cfg['n_layers']:>3}  {cfg['hidden']:>4}  "
              f"{cfg['batch_size']:>5}  {flops:>12,}")
