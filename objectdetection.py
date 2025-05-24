import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import tensorflow as tf
from loguru import logger

from hackathon import objectdetection
from hackathon.config import CONFIG
from hackathon.utils import (
    cast_floating_to,
)

# Ensure TF doesn't consume all GPU memory
tf.config.experimental.set_visible_devices([], "GPU")


def main():
    key = jr.PRNGKey(CONFIG["seed"])

    # Initialize student model
    logger.info("Initializing student...")
    key_backbone, key_decoder = jr.split(key, 2)

    dataset = "voc"
    datasets = ["voc/2007"]
    model = ...
    model = cast_floating_to(model, jnp.float32)
    model_params, model_static = eqx.partition(model, eqx.is_array)

    params = objectdetection.evaluate(
        dataset=datasets,
        params=model_params,
        static=model_static,
        key=key,
        config=CONFIG,
        seed=CONFIG["seed"],
    )


if __name__ == "__main__":
    main()
