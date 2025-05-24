# Object Detection with JAX/Equinox on VOC 2007

This project is a challenge to implement an object detection pipeline for the PASCAL VOC 2007 dataset using JAX and
Equinox. You'll be adapting key components from the YOLOv10 model (originally in PyTorch) to a JAX-based framework.

This project uses JAX, a high-performance numerical computation library, and Equinox, a library for building neural
networks in JAX.

**Primary Goal:** train an object detection model on the VOC 2007 dataset by implementing the prediction head and loss
function inspired by YOLOv10 within the provided JAX/Equinox boilerplate.

## Table of Contents

1. [Prerequisites](#prerequisites)
1. [Project Structure](#project-structure)
1. [Understanding the VOC 2007 Dataset](#understanding-the-voc-2007-dataset)
1. [Your Task: Implementing Object Detection](#your-task-implementing-object-detection)
   - [The Challenge: Porting YOLOv10 Components](#the-challenge-porting-yolov10-components)
   - [Key Files to Modify](#key-files-to-modify)
   - [Guidance on Porting](#guidance-on-porting)
1. [Crash Course: JAX and Equinox for PyTorch Users](#crash-course-jax-and-equinox-for-pytorch-users)
   - [Core Principles](#core-principles)
   - [PRNG Keys: Explicit Randomness](#prng-keys-explicit-randomness)
   - [`jax.jit`: Just-In-Time Compilation](#jaxjit-just-in-time-compilation)
   - [`jax.vmap`: Automatic Vectorization (Batching)](#jaxvmap-automatic-vectorization-batching)
   - [Equinox Modules: `eqx.Module`](#equinox-modules-eqxmodule)
   - [Gradients: `jax.grad` & `jax.value_and_grad`](#gradients-jaxgrad--jaxvalue_and_grad)
   - [PyTrees](#pytrees)
   - [Immutability and Updates](#immutability-and-updates)
1. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
1. [Configuration](#configuration)
1. [Running the Code](#running-the-code)
   - [Setting up the Environment](#setting-up-the-environment)
   - [Running the Classification Example](#running-the-classification-example)
   - [Running Your Object Detection Implementation](#running-your-object-detection-implementation)
1. [Evaluation (for Object Detection)](#evaluation-for-object-detection)
1. [Important Considerations & Debugging Tips](#important-considerations--debugging-tips)
1. [Resources](#resources)

## Prerequisites

- Basic understanding of machine learning and deep learning concepts.
- Experience with Python.
- Familiarity with other DL frameworks like PyTorch is highly beneficial,
- No prior JAX/Equinox experience is strictly necessary,
- Access to a machine with a GPU/TPU is recommended for faster training, though not strictly required.

## Project Structure

The project is organized as follows:

```
.
├── hackathon/
│   ├── init.py
│   ├── augmentations.py       # Image augmentation functions (TensorFlow based)
│   ├── classification.py      # WORKING EXAMPLE: Image classification task
│   ├── config.py              # Configuration loading
│   ├── dataloader.py          # TensorFlow Datasets (TFDS) loading logic
│   ├── metrics.py             # (Potentially) Metrics like Dice coefficient
│   ├── objectdetection.py     # BOILERPLATE: Your primary workspace for object detection
│   ├── preprocessing.py       # Image preprocessing functions (TensorFlow based)
│   ├── tasks.py               # Task-specific processing function dispatcher
│   └── utils.py               # Utility functions
├── classification.py          # Main script to run training/evaluation for a classification task
├── objectdetection.py         # The main script to update to run the object detection pipeline
├── .config                    # Example configuration file (rename from .config.example if provided)
└── README.md                  # This file
```

- **`hackathon/classification.py`**: This file contains a fully working example of an image classification task.
- **`hackathon/objectdetection.py`**: This is where you'll spend most of your time. It contains boilerplate code for an
  object detection task, with several `NotImplementedError` sections that you need to fill in.
- **`augmentations.py` & `preprocessing.py`**: These files use TensorFlow for image operations. Note that the data
  pipeline (loading, preprocessing, augmentation) is TensorFlow-based, and the resulting NumPy arrays are then fed into
  JAX for model training.
- **`config.py`**: Manages project configurations. See the `.config` file for available options.

The base neural network used for feature extraction is a custom Convolutional Neural Network (CNN). Its definition is
another open source project, [Equimo](https://github.com/clementpoiret/Equimo), a project implementing multiple computer
vision models using Equinox.

## Understanding the VOC 2007 Dataset

The PASCAL VOC 2007 dataset is a standard benchmark for object detection. Key characteristics:

- **Images:** Contains thousands of color images of various scenes.
- **Object Classes:** 20 object classes (e.g., person, car, cat, dog, chair).
- **Annotations:** Each image is annotated with:
  - Bounding boxes for every object instance.
  - The class label for each bounding box.
- **Challenges:**
  - Multiple objects per image.
  - Objects of varying sizes and scales.
  - Occlusion and truncation of objects.

Your model will need to predict the class and bounding box coordinates (typically (x\_{min}, y\_{min}, x\_{max}, y\_{max})) for each object in an image.

## The Task: Implementing Object Detection

Your primary task is to complete the `hackathon/objectdetection.py` script. This involves:

1. **Understanding the Data:**

   - Familiarize yourself with how bounding boxes and labels are provided by the `dataloader.py` for the VOC dataset.
     The `preprocessing` function in `objectdetection.py` receives `sample[objects_key]["bbox"]` and
     `sample[objects_key]["label"]`. You'll need to process these appropriately.
   - The `augmentations.py` file contains a `create_global_crops` function that has a `NotImplementedError` section for
     handling bounding boxes during augmentations. You will need to implement this to ensure bounding boxes are
     correctly transformed along with the images. The `random_resized_crop` function already has bbox transformation
     logic you can adapt.

1. **Implementing the YOLOv10-inspired Prediction Head:**

   - In `objectdetection.py`, the `loss_fn` calls `model(...)`. This `model` (your custom CNN + prediction head) will
     output predictions. You need to design and implement a prediction head that takes features from the base CNN and
     outputs predictions in a format suitable for object detection following the Yolov10 network.
   - Refer to the YOLOv10 architecture for inspiration on how the prediction head is structured. You'll need to
     implement these layers using Equinox modules (`eqx.nn.Conv2d`, `eqx.nn.Linear`, etc.). You can check out the Equimo
     repository to have many examples of advanced types of layers.

1. **Implementing the YOLOv10-inspired Loss Function:**

   - The `loss_fn` in `objectdetection.py` currently has a `NotImplementedError`. You need to implement the loss
     calculation. This will likely involve several components:
     - **Classification Loss:** For the class of detected objects.
     - **Regression Loss:** For the bounding box coordinates.
     - **Objectness Loss:** To determine if an anchor/grid cell contains an object.
   - You'll need to match ground truth bounding boxes with predicted boxes to calculate these losses. This is a
     critical part of object detection.

1. **Completing `preprocessing` and `augmentations`:**

   - The `preprocessing` function in `objectdetection.py` needs to be completed to handle bounding box data alongside
     images. This might involve converting bounding box formats or preparing them for augmentation.
   - The `augmentations` function in `objectdetection.py` also needs completion. Crucially, the `create_global_crops`
     call within it needs to correctly handle bounding box transformations. The `bboxes` argument to `create_global_crops`
     (in `augmentations.py`) is where you'll pass your bounding box data, and you'll need to implement the logic within
     `create_global_crops` to transform these bboxes according to the image augmentations.

1. **Implementing Evaluation:**

   - The evaluation loop in `evaluate` within `objectdetection.py` has a `NotImplementedError`. You'll need to
     implement a suitable evaluation metric for object detection, such as mean Average Precision (mAP). This will involve
     processing model predictions on the validation set and comparing them against ground truth.

### The Challenge: Porting YOLOv10 Components

You are **not** expected to port the entire YOLOv10 model. The focus is on the **prediction heads** and the **loss
computation**.

- **YOLOv10 Repository:** [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- Study the PyTorch implementation of the heads and loss functions in the YOLOv10 repository. Your goal is to translate
  the *logic* and *mathematical operations* into JAX/Equinox.

### Key Files to Modify

- **`hackathon/objectdetection.py`**:
  - `preprocessing()`: Implement logic to handle image and object (bounding box, label) data.
  - `augmentations()`: Ensure bounding boxes are correctly passed to and processed by `create_global_crops`.
  - `loss_fn()`: Implement the object detection loss.
  - `evaluate()`: Implement the evaluation loop and mAP calculation.
  - You may also need to define new Equinox modules for the prediction head(s) here or in a separate file.
- **`hackathon/augmentations.py`**:
  - `create_global_crops()`: Implement the bounding box transformation logic when `bboxes` are provided.

### Guidance on Porting

- **Start Simple:** Begin by understanding the structure of a single detection head layer.
- **Equinox Equivalents:**
  - PyTorch `nn.Conv2d` -> `eqx.nn.Conv2d`
  - PyTorch `nn.Linear` -> `eqx.nn.Linear`
  - Activation functions (ReLU, Sigmoid, etc.) are available in `jax.nn`.
- **Shape Management:** Pay very close attention to tensor shapes at each step. JAX is explicit about this, i.e., it
  will not explicitely broadcast shapes like PyTorch.
- **Loss Components:** Break down the YOLO loss into its constituent parts and implement them one by one.

## Crash Course: JAX and Equinox for PyTorch Users

JAX and Equinox have some fundamental differences from PyTorch. Understanding these is crucial.

### Core Principles

- **Functional Programming:** JAX encourages a functional programming style. Functions should ideally be *pure* (no
  side effects; output depends only on input).
- **Immutability:** JAX arrays (and Equinox model parameters) are immutable. Operations that "modify" an array actually
  return a new array.

### PRNG Keys: Explicit Randomness

Unlike PyTorch's global random seed, JAX requires explicit handling of pseudo-random number generator (PRNG) keys.

- Create a key: `key = jax.random.PRNGKey(seed)`
- Split keys for different operations: `key, subkey = jax.random.split(key)`
- Pass keys to functions that require randomness (e.g., initializers, dropout, sampling).
- **Why?** This makes random operations reproducible and easier to reason about, especially with transformations like
  `jax.vmap` and `jax.pmap`.

**Example:**

```python
import jax
import jax.random as jr

key = jr.PRNGKey(0)
key_dropout, key_init, key_data_aug = jr.split(key, 3)

# Use key_dropout for a dropout layer
# Use key_init for weight initialization
# Use key_data_aug for a random data augmentation
```

In our codebase, you'll see `key` being passed around, especially in `train_step` and `compute_cls_accuracy`.

### `jax.jit`: Just-In-Time Compilation

JAX can compile your Python functions into highly optimized XLA (Accelerated Linear Algebra) code using `jax.jit`.

- **Usage:** Decorate your function with `@jax.jit` (or `@eqx.filter_jit` for Equinox modules).
- **Benefits:** Significant speedups, especially for numerical computations.
- **Gotchas:**
  - The function is traced on its first call with specific input shapes and types. Subsequent calls with different shapes/types will trigger a re-compilation.
  - Control flow (if/else, loops) based on tensor *values* can be tricky. Try to make control flow depend on static values or use JAX control flow primitives like `jax.lax.cond` or `jax.lax.scan`.
  - Side effects (like printing or modifying external variables) inside JITted functions behave unexpectedly or are ignored. Use `jax.debug.print` for debugging.

### `jax.vmap`: Automatic Vectorization (Batching)

`jax.vmap` is a powerful transformation that maps a function designed for single data points to work over batches of data.

- **PyTorch:** Batching is often implicit in layers (e.g., `nn.Conv2d` expects a batch dimension).
- **JAX/Equinox:** You typically write functions/modules for a *single* data sample. `jax.vmap` then handles the batching.
  - `jax.vmap(model)(batch_of_images)`
  - `in_axes` argument specifies which arguments to map over (e.g., `in_axes=(0, None)` means map over the first dimension of the first argument, and broadcast the second argument).

**Example from `classification.py`:**

```python
# model is defined to process a single image
logits = jax.vmap(model, in_axes=(0, None))(images, key)
# Here, `images` is a batch of images (e.g., shape [B, H, W, C])
# `key` is a single PRNG key, broadcasted for each image in the batch.
# `model` is applied to each image in `images`.
```

You **must** use `jax.vmap` (or ensure your model inherently handles batches, which is less common in Equinox) when
processing batches of data.

### Equinox Modules: `eqx.Module`

Equinox provides a simple way to create neural network modules, similar to `torch.nn.Module`.

- **Structure:** Parameters (weights, biases) are attributes of the class.
- **`eqx.filter`:** Equinox modules are "PyTrees." `eqx.filter` helps separate learnable parameters from static configuration or non-learnable components.
  - `params, static = eqx.filter(model, eqx.is_array)` separates array attributes (learnable parameters) from everything else.
  - `model = eqx.combine(params, static)` reconstructs the model.
    This is crucial for optimizers, which only need to update the learnable parameters.

**Example from `classification.py`:**

```python
# In train_step:
model = eqx.combine(params, static) # Reconstruct model
# ...
loss, grads = jax.value_and_grad(loss_for_step)(params) # Get grads only for params
# params = optax.apply_updates(params, updates) # Update only params
```

### Gradients: `jax.grad` & `jax.value_and_grad`

JAX provides functions for automatic differentiation.

- `jax.grad(fun)`: Returns a function that computes the gradient of `fun`.
- `jax.value_and_grad(fun)`: Returns a function that computes both the value of `fun` and its gradient. This is often more efficient.

The `loss_fn` in the provided code is designed to take `params` (the learnable parts of the model) as its first argument,
which is what `jax.grad` or `jax.value_and_grad` will differentiate with respect to.

### PyTrees

JAX and Equinox extensively use the concept of "PyTrees." A PyTree is a container of leaf elements, like lists, tuples,
and dictionaries. JAX transformations (`jit`, `vmap`, `grad`) can operate on entire PyTrees. Equinox modules are PyTrees.
This allows you to handle complex nested structures of parameters and data seamlessly.

### Immutability and Updates

Since JAX arrays are immutable, you don't update them in-place.

```python
# PyTorch (in-place)
params[0] += update_val

# JAX (functional update)
params = params.at[0].add(update_val)
# Or, for entire arrays, often just:
params = new_params
```

Optax optimizers handle this correctly when applying updates: `params = optax.apply_updates(params, updates)`.

## Data Loading and Preprocessing

- **`hackathon/dataloader.py`**: Uses `tensorflow_datasets` (TFDS) to load data.
  - The `load_tfds` function handles dataset loading, preprocessing, augmentation, batching, and conversion to NumPy arrays.
- **`hackathon/preprocessing.py`**: Contains TensorFlow-based image preprocessing functions like `prepare_image` (resizing) and `normalize`.
- **`hackathon/augmentations.py`**: Contains TensorFlow-based image augmentation functions like `random_resized_crop`, `random_horizontal_flip`, `color_jitter`, etc.
  - **Crucially for object detection:** The `random_resized_crop` function has logic to transform bounding boxes.
    You will need to ensure this logic is correctly used and potentially adapted when you implement bounding box handling
    in `create_global_crops` and the `augmentations` function in `objectdetection.py`.
- **`hackathon/tasks.py`**: The `get_processing_functions` utility dynamically selects the correct `preprocessing`, `augmentations`, and `postprocessing` functions based on the dataset and task type.

The pipeline is: TFDS -> TensorFlow preprocessing/augmentations -> NumPy arrays -> JAX model.

## Configuration

Project configurations are managed by `hackathon/config.py` using the `everett` library, which reads from environment
variables or a `.config` file.

An example `.config` file might look like:

```ini
seed=42
# Object Detection - VOC
num_classes_voc=20 # Number of classes in VOC
image_size_voc=416 # Example input size for YOLO-like models
batch_size_voc=16
learning_rate_voc=1e-4
warmup_steps_voc=100
num_epochs_voc=50
# ... other VOC specific params ...
# image params
patch_size=16 # If your base CNN is ViT-like
normalization_mean=0.485,0.456,0.406
normalization_std=0.229,0.224,0.225
global_crops_size=224
# ...
```

Make sure your `.config` file has the necessary parameters for `voc` (e.g., `num_classes_voc`, `image_size_voc`,
`batch_size_voc_scratch`, etc.). The provided `config.py` lists all expected configurations.

## Evaluation (for Object Detection)

For object detection, a common evaluation metric is **mean Average Precision (mAP)**.

- You will need to implement mAP calculation in the `evaluate` function of `objectdetection.py`.
- This involves:
  1. Getting predictions from your model on the validation set (bounding boxes, class scores, objectness scores).
  1. Applying Non-Maximum Suppression (NMS) to filter redundant boxes.
  1. Comparing predicted boxes to ground truth boxes using Intersection over Union (IoU).
  1. Calculating precision and recall for each class at various confidence thresholds.
  1. Computing the Average Precision (AP) for each class (area under the precision-recall curve).
  1. Averaging APs across all classes to get mAP.

This is a non-trivial task, so plan your time accordingly. You might find existing JAX implementations of mAP or its
components helpful as a reference.

## Important Considerations & Debugging Tips

- **Start with `classification.py`**: Understand it thoroughly. It's your reference for JAX/Equinox patterns.
- **Incremental Development**: Implement and test small pieces at a time.
  - Can you get the data shapes right for the head?
  - Can you compute one component of the loss?
- **Debugging JITted Code**:
  - `jax.debug.print("message: {x}", x=array)`: Prints values inside JITted functions. It does not print during tracing, only during execution.
  - `jax.debug.breakpoint()`: If you run your code under a Python debugger.
  - Temporarily disable `@jax.jit` or `@eqx.filter_jit` for easier debugging with standard Python print statements, but
    remember performance will be much slower.
- **Shape Mismatches**: This will be a common source of errors. Print shapes frequently!
  - `print(array.shape)` (outside JIT)
  - `jax.debug.print("array shape: {s}", s=array.shape)` (inside JIT)
- **PRNG Key Management**: Ensure keys are split and threaded through your functions correctly. Reusing keys will lead to correlated "random" numbers.
- **Bounding Box Formats**: Be consistent with your bounding box coordinate format (e.g., `[xmin, ymin, xmax, ymax]` vs. `[x_center, y_center, width, height]`, normalized vs. absolute). VOC typically uses `[ymin, xmin, ymax, xmax]` (absolute). TFDS might provide them normalized. Check `dataloader.py` and `preprocessing.py`.
- **Numerical Stability**: Watch out for `NaN`s in your loss. This can be due to log(0), division by zero, or large gradients. Use `jax.numpy.clip` or add small epsilon values (`1e-7` or `jnp.finfo(dtype).eps`) where appropriate.
- **TensorFlow vs. JAX operations**: Remember that the data loading pipeline uses TensorFlow ops. Your model and loss function will use JAX ops (`jax.numpy` which is often aliased as `jnp`).

## Resources

- **JAX Documentation:** [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
- **Equinox Documentation:** [https://docs.kidger.site/equinox/](https://docs.kidger.site/equinox/)
- **Optax (Optimizer library) Documentation:** [https://optax.readthedocs.io/](https://optax.readthedocs.io/)
- **YOLOv10 Paper & Repository:** (Linked above)
- **PASCAL VOC Dataset:** [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
- **TFDS VOC2007:** [https://www.tensorflow.org/datasets/catalog/voc#voc2007](https://www.tensorflow.org/datasets/catalog/voc#voc2007) (Details on how TFDS structures the data)
