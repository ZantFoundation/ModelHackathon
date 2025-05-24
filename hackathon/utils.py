import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def cast_floating_to(tree, dtype: jnp.dtype):
    def conditional_cast(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(
            x.dtype, jnp.floating
        ):
            x = x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


def count_params(model: eqx.Module):
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    return num_params / 1_000_000


def float_list_parser(s: str) -> list[float]:
    return list(map(float, s.split(",")))


def int_list_parser(s: str) -> list[int]:
    return list(map(int, s.split(",")))
