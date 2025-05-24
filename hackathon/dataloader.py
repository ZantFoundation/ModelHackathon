import tensorflow as tf
import tensorflow_datasets as tfds

from hackathon.tasks import get_processing_functions


def load_datasets(datasets: list[str], split: str):
    ds = None
    for dataset_name in datasets:
        current_ds = tfds.load(
            dataset_name,
            split=split,
            as_supervised=False,
            shuffle_files=False,
        )
        if ds is None:
            ds = current_ds
        else:
            ds = ds.concatenate(current_ds)
    return ds


def load_tfds(
    dataset: str | list[str],
    split: str,
    batch_size: int,
    seed: int,
    preproc_kwargs: dict,
    aug_kwargs: dict,
    postproc_kwargs: dict,
):
    rng = tf.random.Generator.from_seed(seed)

    if isinstance(dataset, str):
        dataset_list = [dataset]
    else:
        dataset_list = dataset

    ds = load_datasets(dataset_list, split)

    preprocess, augment, postprocess = get_processing_functions(
        dataset_list[0],  # assuming all datasets in the list need the same functions
        preproc_kwargs=preproc_kwargs,
        aug_kwargs=aug_kwargs,
        postproc_kwargs=postproc_kwargs,
    )

    def seeded_augment(sample):
        seed = rng.make_seeds(1)[:, 0]
        return augment(sample, seed=seed)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        ds = ds.cache()
        ds = ds.map(seeded_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)

    ds = ds.map(postprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if split != "train":
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return tfds.as_numpy(ds)
