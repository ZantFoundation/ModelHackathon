import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow as tf
from einops import reduce
from loguru import logger
from tqdm import tqdm

from hackathon.augmentations import (
    create_global_crops,
    create_local_crops,
)
from hackathon.dataloader import load_tfds
from hackathon.metrics import dice_coefficient
from hackathon.preprocessing import (
    nhwc_to_nchw,
    normalize,
    prepare_image,
)


def preprocessing(
    sample,
    image_size: int,
    num_classes: int,
    is_training: bool = True,
    image_key: str = "image",
    objects_key: str = "objects",
    objects_map_interp: str = "bilinear",
):
    raw_image = sample[image_key]
    raw_bboxes = sample[objects_key]["bbox"]
    raw_labels = sample[objects_key]["label"]

    image = prepare_image(raw_image, image_size)

    raise NotImplementedError

    return {"image": image, "objects": objects}


def augmentations(
    sample: dict,
    num_classes: int,
    seed,
    local_crops: bool = True,
    n_global_crops: int = 2,
    gc_kwargs: dict = {},
    lc_kwargs: dict = {},
):
    seeds = tf.random.split(seed, 2)

    image, bboxes, labels = sample["image"], sample["bboxes"], sample["labels"]

    # global_crops, objects_crops = create_global_crops(
    #     image,
    #     bboxes=bboxes,
    #     crops_number=n_global_crops,
    #     num_classes=num_classes,
    #     labels=labels,
    #     seed=seeds[0],
    #     **gc_kwargs,
    # )

    raise NotImplementedError

    res = {
        "global_crops": global_crops,
        "objects": objects_crops,
    }

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


def get_identifier(name: str | list[str]) -> str:
    if isinstance(name, str):
        return name
    else:
        # assuming we have a list of variants of similar datasets
        # in the format `{dataset}/{variant}`, e.g., `voc/2007`
        return name[0].split("/")[0]


def loss_fn(params, static, images, objects, key):
    model = eqx.combine(params, static)

    logits = jax.vmap(model, in_axes=(0, None, None))(images, key, False)

    loss = None

    raise NotImplementedError

    return jnp.mean(loss)


@eqx.filter_jit
def train_step(params, static, optimizer, opt_state, batch, key):
    images = batch["global_crops"][:, 0, :, :, :]
    objects = batch["objects"][:, 0, :, :, :]
    classes = batch["classes"]

    def loss_for_step(p):
        return loss_fn(p, static, images, objects, classes, key)

    loss, grads = jax.value_and_grad(loss_for_step)(params)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def evaluate(dataset, params, static, key, config, seed):
    dataset_identifier = get_identifier(dataset)

    # Load datasets
    logger.info(f"Loading {dataset_identifier}...")
    train_ds = load_tfds(
        dataset=dataset,
        split="train",
        batch_size=config[f"batch_size_{dataset_identifier}_scratch"],
        preproc_kwargs={
            "image_size": config[f"image_size_{dataset_identifier}"],
            "num_classes": config[f"num_classes_{dataset_identifier}"],
            "is_training": True,
        },
        aug_kwargs={
            "num_classes": config[f"num_classes_{dataset_identifier}"],
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
        batch_size=config[f"batch_size_{dataset_identifier}_scratch"],
        preproc_kwargs={
            "image_size": config[f"image_size_{dataset_identifier}"],
            "num_classes": config[f"num_classes_{dataset_identifier}"],
            "is_training": False,
        },
        aug_kwargs={},  # empty because not used
        postproc_kwargs={},
        seed=seed,
    )

    optimizer = optax.contrib.schedule_free_adamw(
        config[f"learning_rate_{dataset_identifier}_scratch"],
        warmup_steps=config[f"warmup_steps_{dataset_identifier}_scratch"],
    )
    opt_state = optimizer.init(params)

    logger.info(f"Starting evaluation on {dataset_identifier}")
    for epoch in range(config[f"num_epochs_{dataset_identifier}_scratch"]):
        start_time = time.time()
        epoch_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_ds),
            total=len(train_ds),
            desc=f"Epoch {epoch + 1}/{config[f'num_epochs_{dataset_identifier}_scratch']}",
        )
        for batch_idx, batch in progress_bar:
            key, subkey = jr.split(key)
            params, opt_state, loss = train_step(
                params,
                static,
                optimizer,
                opt_state,
                batch,
                key,
            )
            epoch_loss += loss
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed = time.time() - start_time

            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss.item():.4f}",
                    "time": f"{elapsed:.1f}s",
                }
            )

        progress_bar = tqdm(
            val_ds,
            total=len(val_ds),
            desc=f"Epoch {epoch + 1}/{config[f'num_epochs_{dataset_identifier}_scratch']}",
        )
        for batch in progress_bar:
            key, subkey = jr.split(key)
            # Test on a given metric on the validation set (e.g. mAP)
            raise NotImplementedError

    logger.info("Evaluation completed!")

    return params

def loss_fn(preds, targets, num_classes, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
    """
    preds: (B, H, W, A * (5 + C)) raw outputs
    targets: dict with keys:
        - "object_mask": (B, H, W, 1) → 1 if object exists
        - "bbox_target": (B, H, W, 4)
        - "class_target": (B, H, W, C)
    """

    obj_mask = targets["object_mask"]
    bbox_target = targets["bbox_target"]
    class_target = targets["class_target"]

    pred_bbox = preds[..., :4]
    pred_obj = preds[..., 4:5]
    pred_cls = preds[..., 5:]

 
    bbox_loss = jnp.abs(pred_bbox - bbox_target) * obj_mask
    bbox_loss = jnp.sum(bbox_loss) / jnp.sum(obj_mask + 1e-6)


    obj_loss = optax.sigmoid_binary_cross_entropy(pred_obj, obj_mask)
    obj_loss = jnp.mean(obj_loss)

    
    cls_loss = optax.sigmoid_binary_cross_entropy(pred_cls, class_target)
    cls_loss = jnp.sum(cls_loss * obj_mask) / jnp.sum(obj_mask + 1e-6)

    total_loss = lambda_box * bbox_loss + lambda_obj * obj_loss + lambda_cls * cls_loss

    return total_loss, {
        "total_loss": total_loss,
        "bbox_loss": bbox_loss,
        "obj_loss": obj_loss,
        "cls_loss": cls_loss
    }

