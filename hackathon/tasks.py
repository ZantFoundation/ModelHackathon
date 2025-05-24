from functools import partial
from typing import Literal

type KnownDataset = Literal["imagenette", "voc", "voc/2007", "voc/2012"]
type TaskType = Literal[
    "classification", "object_detection", "segmentation", "depth_estimation"
]
TASK_TYPE: dict[KnownDataset, TaskType] = {
    "imagenette": "classification",
    "voc": "object_detection",
    "voc/2007": "object_detection",
    "voc/2012": "object_detection",
}


def get_processing_functions(
    dataset: KnownDataset,
    preproc_kwargs: dict,
    aug_kwargs: dict,
    postproc_kwargs: dict,
):
    task_type = TASK_TYPE[dataset]

    match task_type:
        case "classification":
            from hackathon.classification import (
                preprocessing,
                augmentations,
                postprocessing,
            )

            return (
                partial(preprocessing, **preproc_kwargs),
                partial(augmentations, **aug_kwargs),
                partial(postprocessing, **postproc_kwargs),
            )
        case "object_detection":
            from hackathon.objectdetection import (
                preprocessing,
                augmentations,
                postprocessing,
            )

            return (
                partial(preprocessing, **preproc_kwargs),
                partial(augmentations, **aug_kwargs),
                partial(postprocessing, **postproc_kwargs),
            )
        case "segmentation":
            raise NotImplementedError
        case "depth_estimation":
            raise NotImplementedError
