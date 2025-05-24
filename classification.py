import equimo.models as em
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import tensorflow as tf
from loguru import logger

from hackathon import classification
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

    # Classification benchmarks
    dataset = "imagenette"

    model = em.reduceformer_backbone_b1(
        in_channels=3,
        dropout=CONFIG["student_dpr"],
        drop_path_rate=CONFIG["student_dpr"],
        num_classes=CONFIG[f"num_classes_{dataset}"],
        key=key_backbone,
    )
    model = cast_floating_to(model, jnp.float32)
    model_params, model_static = eqx.partition(model, eqx.is_array)

    classification.evaluate(
        dataset=dataset,
        params=model_params,
        static=model_static,
        key=key,
        config=CONFIG,
        seed=CONFIG["seed"],
    )

    logger.info("All evaluations completed!")


if __name__ == "__main__":
    main()
