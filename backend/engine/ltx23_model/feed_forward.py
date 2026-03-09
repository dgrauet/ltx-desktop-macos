"""Feed-forward networks for LTX-2.3 Transformer.

Adapted from Acelogic/LTX-2-MLX. Key naming changed to match our
converted weight format: proj_in/proj_out instead of project_in/project_out.
"""

import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Architecture: Linear -> GELU -> Linear

    Weight keys: ff.proj_in.{weight,bias}, ff.proj_out.{weight,bias}
    """

    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu_approx(self.proj_in(x))
        return self.proj_out(x)
