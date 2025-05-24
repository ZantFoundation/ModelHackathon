import tensorflow as tf


@tf.function
def nhwc_to_nchw(image: tf.Tensor) -> tf.Tensor:
    if len(image.shape) == 4:
        return tf.transpose(image, [0, 3, 1, 2])
    elif len(image.shape) == 3:
        return tf.transpose(image, [2, 0, 1])
    else:
        raise NotImplementedError


@tf.function
def normalize(image, mean, std):
    """
    Normalizes an image tensor of any rank, assuming channels are the last dimension.

    Args:
        image: Input image tensor (any rank, channel-last).
               Expected to have integer values in the range [0, 255].
        mean: List, tuple, or 1D tensor of channel means.
        std: List, tuple, or 1D tensor of channel standard deviations.

    Returns:
        Normalized image tensor (tf.float32).
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0

    img_mean = tf.constant(mean, dtype=tf.float32)
    img_std = tf.constant(std, dtype=tf.float32)

    rank = tf.rank(image)
    num_channels = tf.shape(image)[-1]
    broadcast_shape = tf.concat(
        [tf.ones(rank - 1, dtype=tf.int32), [num_channels]], axis=0
    )

    img_mean = tf.reshape(img_mean, broadcast_shape)
    img_std = tf.reshape(img_std, broadcast_shape)

    normalized_image = (image - img_mean) / img_std
    return normalized_image


@tf.function
def prepare_image(
    image: tf.Tensor,
    image_size: int,
    method: str = "bilinear",
    enforce_last: int | None = 3,
):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size], method=method)

    if enforce_last is not None:
        image = image[:, :, :enforce_last]

    return image
