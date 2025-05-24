from everett.manager import ConfigManager

from hackathon.utils import (
    float_list_parser,
    int_list_parser,
)

cm = ConfigManager.basic_config(env_file=".config")

CONFIG = {
    "seed": cm("seed", parser=int),
    # teacher
    "teacher_cls": cm("teacher_cls", parser=str),
    "teacher_identifier": cm("teacher_identifier", parser=str),
    "teacher_feature_dim": cm("teacher_feature_dim", parser=int),
    # student
    "student_feature_dim": cm("student_feature_dim", parser=int),
    "student_depths": cm("student_depths", parser=int_list_parser),
    "student_num_heads": cm("student_num_heads", parser=int_list_parser),
    "student_init_values": cm("student_init_values", parser=float),
    "student_dropout": cm("student_dropout", parser=float),
    "student_dpr": cm("student_dpr", parser=float),
    # training
    "checkpoint_dir": cm("checkpoint_dir", parser=str),
    "max_to_keep": cm("max_to_keep", parser=int),
    ## Knowledge Distillation
    "dataset_kd": cm("dataset_kd", parser=str),
    "batch_size_kd": cm("batch_size_kd", parser=int),
    "learning_rate_kd": cm("learning_rate_kd", parser=float),
    "warmup_steps_kd": cm("warmup_steps_kd", parser=int),
    "num_epochs_kd": cm("num_epochs_kd", parser=int),
    ## Downtream tasks
    ### Classification
    "num_classes_imagenette": cm("num_classes_imagenette", parser=int),
    "image_size_imagenette": cm("image_size_imagenette", parser=int),
    "batch_size_imagenette": cm("batch_size_imagenette", parser=int),
    "learning_rate_imagenette": cm("learning_rate_imagenette", parser=float),
    "warmup_steps_imagenette": cm("warmup_steps_imagenette", parser=int),
    "num_epochs_imagenette": cm("num_epochs_imagenette", parser=int),
    "batch_size_imagenette_scratch": cm("batch_size_imagenette_scratch", parser=int),
    "learning_rate_imagenette_scratch": cm(
        "learning_rate_imagenette_scratch", parser=float
    ),
    "warmup_steps_imagenette_scratch": cm(
        "warmup_steps_imagenette_scratch", parser=int
    ),
    "num_epochs_imagenette_scratch": cm("num_epochs_imagenette_scratch", parser=int),
    ### Object Detection
    "num_classes_voc": cm("num_classes_voc", parser=int),
    "image_size_voc": cm("image_size_voc", parser=int),
    "batch_size_voc": cm("batch_size_voc", parser=int),
    "learning_rate_voc": cm("learning_rate_voc", parser=float),
    "warmup_steps_voc": cm("warmup_steps_voc", parser=int),
    "num_epochs_voc": cm("num_epochs_voc", parser=int),
    "batch_size_voc_scratch": cm("batch_size_voc_scratch", parser=int),
    "learning_rate_voc_scratch": cm("learning_rate_voc_scratch", parser=float),
    "warmup_steps_voc_scratch": cm("warmup_steps_voc_scratch", parser=int),
    "num_epochs_voc_scratch": cm("num_epochs_voc_scratch", parser=int),
    # images
    "patch_size": cm("patch_size", parser=int),
    "normalization_mean": cm("normalization_mean", parser=float_list_parser),
    "normalization_std": cm("normalization_std", parser=float_list_parser),
    "global_crops_size": cm("global_crops_size", parser=int),
    "global_crops_scale": cm("global_crops_scale", parser=float_list_parser),
    "local_crops_size": cm("local_crops_size", parser=int),
    "local_crops_scale": cm("local_crops_scale", parser=float_list_parser),
    "local_crops_number": cm("local_crops_number", parser=int),
    "mask_ratio_min_max": cm("mask_ratio_min_max", parser=float_list_parser),
    "mask_sample_probability": cm("mask_sample_probability", parser=float),
    # logging
    "log_every_n": cm("log_every_n", parser=int),
}
