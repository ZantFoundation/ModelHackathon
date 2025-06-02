from typing import List, Optional, Tuple, Union

import tensorflow as tf


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.

    4D image => [N, H, W, C] or [N, C, H, W]
    3D image => [1, H, W, C] or [1, C, H, W]
    2D image => [1, H, W, 1]

    Args:
      image: The 2/3/4D input tensor.update-global-crops-bboxes

    Returns:
      A 4D image tensor.

    Raises:
      `TypeError` if `image` is not a 2/3/4D tensor.

    """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def _pad(
    image: tf.Tensor,
    filter_shape: Union[List[int], Tuple[int, ...]],
    mode: str = "CONSTANT",
    constant_values: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    """Explicitly pads a 4-D image.

    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.

    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height and
        width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of
        padding algorithm to use, which is compatible with `mode` argument in
        `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT" padding
        mode.

    Returns:
      A padded image.
    """
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


def _get_gaussian_kernel(sigma: tf.Tensor, filter_size: tf.Tensor) -> tf.Tensor:
    """Computes 1D Gaussian kernel."""
    x = tf.range(
        tf.cast(-filter_size // 2 + 1, tf.float32),
        tf.cast(filter_size // 2 + 1, tf.float32),
    )
    x = tf.cast(x**2, sigma.dtype)
    x = tf.nn.softmax(-x / (2.0 * tf.cast(sigma**2, x.dtype)))
    return x


def _get_gaussian_kernel_2d(
    gaussian_filter_x: tf.Tensor, gaussian_filter_y: tf.Tensor
) -> tf.Tensor:
    """Computes 2D Gaussian kernel given 1D kernels."""
    return tf.matmul(gaussian_filter_y[:, tf.newaxis], gaussian_filter_x[tf.newaxis, :])


@tf.function
def gaussian_filter2d(
    image: tf.Tensor,
    filter_shape: int = 3,
    sigma: float = 1.0,
    padding: str = "REFLECT",
    constant_values: Union[int, float] = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Performs Gaussian blur on image(s)."""
    with tf.name_scope(name or "gaussian_filter2d"):
        # Convert inputs to tensors
        image = tf.convert_to_tensor(image)

        # Handle filter_shape
        filter_h = filter_w = filter_shape

        # Handle sigma
        sigma_h = sigma_w = sigma

        # Input validation
        tf.debugging.assert_greater_equal(sigma_h, 0.0, "sigma_h must be >= 0")
        tf.debugging.assert_greater_equal(sigma_w, 0.0, "sigma_w must be >= 0")

        # Convert image to 4D
        original_ndims = tf.rank(image)
        image = to_4d(image)

        # Handle dtype
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)

        # Get number of channels
        channels = tf.shape(image)[3]

        # Create gaussian kernels
        sigma_t = tf.convert_to_tensor([sigma_h, sigma_w], dtype=image.dtype)

        gaussian_kernel_x = _get_gaussian_kernel(
            sigma_t[1], tf.cast(filter_w, tf.int32)
        )
        gaussian_kernel_y = _get_gaussian_kernel(
            sigma_t[0], tf.cast(filter_h, tf.int32)
        )

        # Create 2D kernel
        gaussian_kernel_2d = _get_gaussian_kernel_2d(
            gaussian_kernel_x, gaussian_kernel_y
        )

        # Reshape kernel for depthwise conv
        gaussian_kernel_2d = tf.reshape(gaussian_kernel_2d, [filter_h, filter_w, 1, 1])
        gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

        # Pad image
        image = _pad(
            image, [filter_h, filter_w], mode=padding, constant_values=constant_values
        )

        # Apply convolution
        output = tf.nn.depthwise_conv2d(
            input=image,
            filter=gaussian_kernel_2d,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        # Convert back to original dimensions and dtype
        output = from_4d(output, original_ndims)
        return tf.cast(output, orig_dtype)


@tf.function
def random_resized_crop(
    image,
    size,
    seed,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    interpolation="bilinear",
    bboxes=None,
):
    """
    Random resized crop for tensorflow, similar to torchvision's RandomResizedCrop.

    Args:
        image: 3D tensor of shape [height, width, channels]
        size: tuple of (height, width) or int for square size
        scale: tuple of (min_scale, max_scale) for area ratio
        ratio: tuple of (min_ratio, max_ratio) for aspect ratio
        interpolation: interpolation method, one of 'bilinear', 'nearest', 'bicubic'
        bboxes: Optional tensor of shape [N, 4] with normalized coordinates [x_min, y_min, x_max, y_max]

    Returns:
        Cropped and resized image tensor
    """
    seeds = tf.random.split(seed, 4)

    if isinstance(size, int):
        size = (size, size)

    # Get image shape
    original_height = tf.cast(tf.shape(image)[0], tf.float32)
    original_width = tf.cast(tf.shape(image)[1], tf.float32)
    original_area = original_height * original_width

    # Get random area ratio
    target_area = original_area * tf.random.stateless_uniform(
        [], minval=scale[0], maxval=scale[1], seed=seeds[0]
    )

    # Get random aspect ratio
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    aspect_ratio = tf.math.exp(
        tf.random.stateless_uniform(
            [], minval=log_ratio[0], maxval=log_ratio[1], seed=seeds[1]
        )
    )

    # Calculate target height and width
    height = tf.sqrt(target_area / aspect_ratio)
    width = height * aspect_ratio

    # Clip dimensions to image size
    height = tf.minimum(height, original_height)
    width = tf.minimum(width, original_width)

    # Get random crop coordinates
    height_int = tf.cast(height, tf.int32)
    width_int = tf.cast(width, tf.int32)

    max_x = tf.cast(original_width - width, tf.int32)
    max_y = tf.cast(original_height - height, tf.int32)

    x = tf.random.stateless_uniform(
        [], minval=0, maxval=max_x + 1, dtype=tf.int32, seed=seeds[2]
    )
    y = tf.random.stateless_uniform(
        [], minval=0, maxval=max_y + 1, dtype=tf.int32, seed=seeds[3]
    )

    # Crop the image
    crop = tf.image.crop_to_bounding_box(image, y, x, height_int, width_int)

    # Resize to target size
    resized = tf.image.resize(crop, size, method=interpolation, antialias=True)

    if bboxes is not None:
        bboxes = tf.cast(bboxes, tf.float32)
        y_float = tf.cast(y, tf.float32)
        x_float = tf.cast(x, tf.float32)
        crop_height = tf.cast(height_int, tf.float32)
        crop_width = tf.cast(width_int, tf.float32)

        # Denormalize bboxes to original image coordinates
        ymin_abs = bboxes[:, 0] * original_height
        xmin_abs = bboxes[:, 1] * original_width
        ymax_abs = bboxes[:, 2] * original_height
        xmax_abs = bboxes[:, 3] * original_width

        # Calculate intersection with the crop box
        intersect_ymin = tf.maximum(ymin_abs, y_float)
        intersect_xmin = tf.maximum(xmin_abs, x_float)
        intersect_ymax = tf.minimum(ymax_abs, y_float + crop_height)
        intersect_xmax = tf.minimum(xmax_abs, x_float + crop_width)

        # Adjust coordinates relative to the crop box origin
        adjusted_ymin = intersect_ymin - y_float
        adjusted_xmin = intersect_xmin - x_float
        adjusted_ymax = intersect_ymax - y_float
        adjusted_xmax = intersect_xmax - x_float

        # Normalize coordinates relative to the crop box dimensions
        # These normalized coordinates are valid for the final resized image
        norm_ymin = adjusted_ymin / crop_height
        norm_xmin = adjusted_xmin / crop_width
        norm_ymax = adjusted_ymax / crop_height
        norm_xmax = adjusted_xmax / crop_width

        # Filter out boxes that are completely outside the crop or have zero area
        valid_indices = tf.where(
            (intersect_xmax > intersect_xmin) & (intersect_ymax > intersect_ymin)
        )

        # Gather valid boxes
        transformed_bboxes = tf.gather_nd(
            tf.stack([norm_ymin, norm_xmin, norm_ymax, norm_xmax], axis=-1),
            valid_indices,
        )

        # Clip final coordinates to [0, 1] range
        transformed_bboxes = tf.clip_by_value(transformed_bboxes, 0.0, 1.0)

        return resized, transformed_bboxes, valid_indices

    return resized


@tf.function
def random_horizontal_flip(image, seed, p=0.5, bboxes=None):
    """
    Randomly flips an image horizontally with a given probability.
    Optionally transforms bounding boxes as well.

    Args:
        image: 3D tensor of shape [height, width, channels].
        p: Float, the probability of flipping the image.
        seed: tf.Tensor seed for stateless random operations.
        bboxes: Optional Nx4 tensor of bounding boxes [ymin, xmin, ymax, xmax]
                in normalized coordinates (relative to the image).

    Returns:
        If bboxes is None:
            The (potentially) flipped image tensor.
        If bboxes is not None:
            A tuple of (the (potentially) flipped image tensor,
                        the (potentially) transformed bboxes tensor).
    """
    should_flip = tf.random.stateless_uniform([], seed=seed) < p

    # Flip image if needed
    flipped_image = tf.cond(
        should_flip, lambda: tf.image.flip_left_right(image), lambda: image
    )

    if bboxes is None:
        return flipped_image
    else:
        bboxes = tf.cast(bboxes, tf.float32)

        # If flipping, transform bounding boxes
        # Original: [ymin, xmin, ymax, xmax]
        # Flipped:  [ymin, 1 - xmax, ymax, 1 - xmin]
        def flip_bboxes(b):
            ymin, xmin, ymax, xmax = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            flipped_xmin = 1.0 - xmax
            flipped_xmax = 1.0 - xmin
            return tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)

        transformed_bboxes = tf.cond(
            should_flip, lambda: flip_bboxes(bboxes), lambda: bboxes
        )
        return flipped_image, transformed_bboxes


@tf.function
def color_jitter(
    image: tf.Tensor,
    seed,
    brightness: Optional[float] = 0.0,
    contrast: Optional[float] = 0.0,
    saturation: Optional[float] = 0.0,
    hue: Optional[float] = 0.0,
    p_grayscale: Optional[float] = 0.0,
) -> tf.Tensor:
    """Applies color jitter to an image, similarly to torchvision`s ColorJitter.

    Args:
      image (tf.Tensor): Of shape [height, width, 3] and type uint8.
      brightness (float, optional): Magnitude for brightness jitter. Defaults to
        0.
      contrast (float, optional): Magnitude for contrast jitter. Defaults to 0.
      saturation (float, optional): Magnitude for saturation jitter. Defaults to
        0.
      hue (float, optional): Magnitude for hue jitter. Defaults to 0.
      p_grayscale (float, optional): Probability to convert the image in grayscale. Defaults to 0.
      seed (int, optional): Random seed. Defaults to None.

    Returns:
      tf.Tensor: The augmented `image` of type uint8.
    """
    seeds = tf.random.split(seed, 5)

    def apply_color_jitter():
        img = tf.image.stateless_random_brightness(image, brightness, seed=seeds[0])
        img = tf.image.stateless_random_contrast(
            img, 1 - contrast, 1 + contrast, seed=seeds[1]
        )
        img = tf.image.stateless_random_saturation(
            img, 1 - saturation, 1 + saturation, seed=seeds[2]
        )
        img = tf.image.stateless_random_hue(img, hue, seed=seeds[3])
        return img

    def apply_grayscale():
        img = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(img)

    random_value = tf.random.stateless_uniform([], seed=seeds[4])

    return tf.cond(random_value < p_grayscale, apply_grayscale, apply_color_jitter)


@tf.function
def gaussian_blur(
    image: tf.Tensor,
    seed,
    p: float = 0.5,
) -> tf.Tensor:
    seeds = tf.random.split(seed, 2)

    random_value = tf.random.stateless_uniform([], seed=seeds[0])
    sigma = tf.random.stateless_uniform([], minval=0.1, maxval=2.0, seed=seeds[1])
    blurred = gaussian_filter2d(image, filter_shape=9, sigma=sigma)

    should_blur = tf.less(random_value, p)
    return tf.cond(should_blur, lambda: blurred, lambda: image)


@tf.function
def solarize(
    image: tf.Tensor,
    seed,
    p: float = 0.2,
    normalized_image: bool = False,  # i.e., pixels \in [0,1]
) -> tf.Tensor:
    random_value = tf.random.stateless_uniform([], seed=seed)

    should_solarize = tf.less(random_value, p)

    threshold = 0.5 if normalized_image else 127.5
    max_value = 1.0 if normalized_image else 255.0

    solarized = tf.where(image < threshold, image, max_value - image)

    return tf.cond(should_solarize, lambda: solarized, lambda: image)


@tf.function
def create_global_crops(
    image: tf.Tensor,
    crops_number: int,
    size: Union[Tuple[int], int],
    scale: Tuple[float, float],
    seed,
    image_label: tf.Tensor | None = None,
    image_label_interp: str = "nearest",
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1,
    p_grayscale: float = 0.2,
    p_gaussian_blur: float = 1.0,
    p_solarize: float = 0.2,
    bboxes: tf.Tensor | None = None,
    labels: tf.Tensor | None = None,
    num_classes: int | None = None,
) -> tf.Tensor | Tuple[tf.Tensor, ...]:
    if bboxes is not None:
        if labels is None or num_classes is None:
            raise ValueError(
                "You must provide `labels` and `num_classes` when passing `bboxes`."
            )
    seed_crops = tf.random.split(seed, crops_number)
    should_transform_label = image_label is not None

    crops = []
    crops_heatmaps = []
    label_crops = [] if should_transform_label else None
    for i in range(crops_number):
        seeds = tf.random.split(seed_crops[i], 5)

        if bboxes is not None:
            crop, crop_bboxes, valid_indices = random_resized_crop(
                image,
                size=size,
                scale=scale,
                bboxes=bboxes,
                seed=seeds[0],
            )
            crop, crop_bboxes = random_horizontal_flip(
                crop, seed=seeds[1], bboxes=crop_bboxes
            )
            crop_labels = tf.gather(labels, valid_indices[:, 0])

            crops_heatmaps.append({
                "bboxes": crop_bboxes,
                "labels": crop_labels
            })
        else:
            crop = random_resized_crop(image, size=size, scale=scale, seed=seeds[0])
            crop = random_horizontal_flip(crop, seed=seeds[1])

        crop = color_jitter(
            crop,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p_grayscale=p_grayscale,
            seed=seeds[2],
        )
        crop = gaussian_blur(crop, p=p_gaussian_blur, seed=seeds[3])
        crop = solarize(crop, p=p_solarize, seed=seeds[4])

        crops.append(crop)
        if bboxes is not None:
            raise NotImplementedError

        if should_transform_label:
            label = random_resized_crop(
                image_label,
                size=size,
                scale=scale,
                interpolation=image_label_interp,
                seed=seeds[0],
            )
            label = tf.image.stateless_random_flip_left_right(label, seed=seeds[1])

            label_crops.append(label)

    crops = tf.stack(crops)

    if bboxes is not None:
        return crops, tf.stack(crops_heatmaps)

    if should_transform_label:
        return crops, tf.stack(label_crops)

    return crops


@tf.function
def create_local_crops(
    image: tf.Tensor,
    crops_number: int,
    size: Union[Tuple[int, int], int],
    scale: Tuple[float, float],
    seed,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1,
    p_grayscale: float = 0.2,
) -> tf.Tensor:
    seed_crops = tf.random.split(seed, crops_number)

    crops = []
    for i in range(crops_number):
        seeds = tf.random.split(seed_crops[i], 4)

        crop = random_resized_crop(image, size=size, scale=scale, seed=seeds[0])
        crop = tf.image.stateless_random_flip_left_right(crop, seed=seeds[1])
        crop = color_jitter(
            crop,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p_grayscale=p_grayscale,
            seed=seeds[2],
        )
        crop = gaussian_blur(crop, p=0.5, seed=seeds[3])
        crops.append(crop)
    return tf.stack(crops)
