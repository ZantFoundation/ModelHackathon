import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow as tf
from loguru import logger
from optax.losses import softmax_cross_entropy
from tqdm import tqdm

from hackathon.augmentations import (
    create_global_crops,
    create_local_crops,
)
from hackathon.dataloader import load_tfds
from hackathon.preprocessing import (
    nhwc_to_nchw,
    normalize,
    prepare_image,
)


def preprocessing(
    sample,
    image_size: int,
    image_label: str = "image",
):
    image = prepare_image(sample[image_label], image_size)

    return sample | {image_label: image}


def augmentations(
    sample: dict,
    seed,
    local_crops: bool = True,
    n_global_crops: int = 2,
    gc_kwargs: dict = {},
    lc_kwargs: dict = {},
):
    seeds = tf.random.split(seed, 2)

    image, label = sample["image"], sample["label"]

    res = {"label": label}

    res["global_crops"] = create_global_crops(
        image, crops_number=n_global_crops, seed=seeds[0], **gc_kwargs
    )

    if local_crops:
        res["local_crops"] = create_local_crops(image, seed=seeds[1], **lc_kwargs)

    return res


def postprocessing(
    sample,
    normalize_image: bool = True,
    normalization_params: tuple | None = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),  # imagenet
    permute_image: bool = True,
    image_labels: list[str] = ["image", "images", "global_crops", "local_crops"],
):
    res = sample
    for label in image_labels:
        if label not in sample:
            continue
        image = sample[label]

        if normalize_image:
            if normalization_params is None:
                raise ValueError(
                    "`normalization_params` needs to be provided if `normalize_image` is True"
                )
            image = normalize(image, *normalization_params)

        if permute_image:
            image = nhwc_to_nchw(image)

        res = res | {label: image}

    return res


def loss_fn(params, static, images, labels, num_classes, key):
    model = eqx.combine(params, static)

    logits = jax.vmap(model, in_axes=(0, None))(images, key)

    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    loss = softmax_cross_entropy(logits, one_hot_labels)

    return jnp.mean(loss)


@eqx.filter_jit
def compute_cls_accuracy(params, static, batch, key):
    images = batch["image"]
    labels = batch["label"]

    model = eqx.combine(params, static)

    # Order of arguments are: image, prngkey, inference bool
    logits = jax.vmap(model, in_axes=(0, None, None))(images, key, True)

    predictions = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy


@eqx.filter_jit
def train_step(params, static, optimizer, opt_state, batch, num_classes, key):
    images = batch["global_crops"][:, 1, :, :, :]
    labels = batch["label"]

    def loss_for_step(p):
        return loss_fn(p, static, images, labels, num_classes, key)

    loss, grads = jax.value_and_grad(loss_for_step)(params)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def evaluate(dataset, params, static, key, config, seed):
    # Load datasets
    logger.info(f"Loading {dataset}...")
    train_ds = load_tfds(
        dataset=dataset,
        split="train",
        batch_size=config[f"batch_size_{dataset}_scratch"],
        preproc_kwargs={"image_size": config[f"image_size_{dataset}"]},
        aug_kwargs={
            "local_crops": False,
            "n_global_crops": 1,
            "gc_kwargs": {
                "size": config["global_crops_size"],
                "scale": config["global_crops_scale"],
            },
        },
        postproc_kwargs={},
        seed=seed,
    )
    val_ds = load_tfds(
        dataset=dataset,
        split="validation",
        batch_size=config[f"batch_size_{dataset}_scratch"],
        preproc_kwargs={"image_size": config[f"image_size_{dataset}"]},
        aug_kwargs={},  # empty because not used
        postproc_kwargs={},
        seed=seed,
    )

    optimizer = optax.contrib.schedule_free_adamw(
        config[f"learning_rate_{dataset}_scratch"],
        warmup_steps=config[f"warmup_steps_{dataset}_scratch"],
    )
    opt_state = optimizer.init(params)

    logger.info(f"Starting evaluation on {dataset}")
    for epoch in range(config[f"num_epochs_{dataset}_scratch"]):
        start_time = time.time()
        epoch_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_ds),
            total=len(train_ds),
            desc=f"Epoch {epoch + 1}/{config[f'num_epochs_{dataset}_scratch']}",
        )
        for batch_idx, batch in progress_bar:
            key, subkey = jr.split(key)
            params, opt_state, loss = train_step(
                params,
                static,
                optimizer,
                opt_state,
                batch,
                config[f"num_classes_{dataset}"],
                key,
            )
            epoch_loss += loss
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed = time.time() - start_time

            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "time": f"{elapsed:.1f}s",
                }
            )

        # Evaluate on test set
        test_accuracy = 0.0
        test_batch_count = 0

        progress_bar = tqdm(
            val_ds,
            total=len(val_ds),
            desc=f"Epoch {epoch + 1}/{config[f'num_epochs_{dataset}_scratch']}",
        )
        for batch in progress_bar:
            key, subkey = jr.split(key)
            batch_accuracy = compute_cls_accuracy(params, static, batch, subkey)
            test_accuracy += batch_accuracy
            test_batch_count += 1

        avg_test_accuracy = test_accuracy / test_batch_count

    logger.info("Evaluation completed!")

    return params
