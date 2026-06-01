from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import albumentations as A
import numpy as np
from PIL import Image
from torchvision import transforms


def _to_min_max(value: Any) -> Any:
    if isinstance(value, dict) and "min" in value and "max" in value:
        return (value["min"], value["max"])
    return value


def _normalize_std_range(std_range: Any) -> Any:
    range_tuple = _to_min_max(std_range)
    if (
        isinstance(range_tuple, tuple)
        and len(range_tuple) == 2
        and all(isinstance(v, (int, float)) for v in range_tuple)
        and max(range_tuple) > 1.0
    ):
        return (range_tuple[0] / 255.0, range_tuple[1] / 255.0)
    return range_tuple


@dataclass(frozen=True)
class AugmentationSpec:
    name: str
    build: Callable[[dict[str, Any]], A.BasicTransform]


class AugmentationRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, AugmentationSpec] = {}
        self._register_defaults()

    def register(self, spec: AugmentationSpec) -> None:
        self._registry[spec.name] = spec

    def build_transform(self, name: str, params: Any) -> A.BasicTransform | None:
        spec = self._registry.get(name)
        if spec is None:
            return None
        if params is True:
            params = {}
        if params in (False, None):
            return None
        if not isinstance(params, dict):
            params = {"value": params}
        return spec.build(params)

    def _register_defaults(self) -> None:
        self.register(
            AugmentationSpec(
                "affine",
                lambda cfg: A.Affine(
                    scale=_to_min_max(cfg.get("scale", {"min": 1.0, "max": 1.0})),
                    translate_percent=_to_min_max(
                        cfg.get("translate_percent", {"min": 0.0, "max": 0.0})
                    ),
                    rotate=_to_min_max(cfg.get("rotate", {"min": 0, "max": 0})),
                    shear=_to_min_max(cfg.get("shear", {"min": 0, "max": 0})),
                    interpolation=cfg.get("interpolation", 1),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "brightness_contrast",
                lambda cfg: A.RandomBrightnessContrast(
                    brightness_limit=_to_min_max(
                        cfg.get("brightness_limit", {"min": -0.2, "max": 0.2})
                    ),
                    contrast_limit=_to_min_max(
                        cfg.get("contrast_limit", {"min": -0.2, "max": 0.2})
                    ),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "clahe",
                lambda cfg: A.CLAHE(
                    clip_limit=cfg.get("clip_limit", 4.0),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(AugmentationSpec("coarse_dropout", self._build_coarse_dropout))
        self.register(
            AugmentationSpec(
                "color_jitter",
                lambda cfg: A.ColorJitter(
                    brightness=_to_min_max(cfg.get("brightness", {"min": 0.8, "max": 1.2})),
                    contrast=_to_min_max(cfg.get("contrast", {"min": 0.8, "max": 1.2})),
                    saturation=_to_min_max(cfg.get("saturation", {"min": 0.8, "max": 1.2})),
                    hue=cfg.get("hue", 0.0),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "defocus",
                lambda cfg: A.Defocus(
                    radius=_to_min_max(cfg.get("radius", {"min": 3, "max": 10})),
                    alias_blur=_to_min_max(
                        cfg.get("alias_blur", {"min": 0.1, "max": 0.5})
                    ),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "gauss_noise",
                lambda cfg: A.GaussNoise(
                    std_range=_normalize_std_range(
                        cfg.get("std_range", {"min": 0.05, "max": 0.15})
                    ),
                    mean_range=_to_min_max(cfg.get("mean_range", {"min": 0.0, "max": 0.0})),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "horizontal_flip",
                lambda cfg: A.HorizontalFlip(p=cfg.get("p", 0.5)),
            )
        )
        self.register(
            AugmentationSpec(
                "iso_noise",
                lambda cfg: A.ISONoise(
                    color_shift=_to_min_max(
                        cfg.get("color_shift", {"min": 0.01, "max": 0.05})
                    ),
                    intensity=_to_min_max(cfg.get("intensity", {"min": 0.1, "max": 0.5})),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(AugmentationSpec("random_resized_crop", self._build_random_resized_crop))
        self.register(
            AugmentationSpec(
                "random_rotate_90",
                lambda cfg: A.RandomRotate90(p=cfg.get("p", 0.5)),
            )
        )
        self.register(
            AugmentationSpec(
                "random_shadow",
                lambda cfg: A.RandomShadow(
                    num_shadows_limit=_to_min_max(
                        cfg.get("num_shadows_limit", {"min": 1, "max": 2})
                    ),
                    shadow_dimension=cfg.get("shadow_dimension", 5),
                    p=cfg.get("p", 0.5),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "resize",
                lambda cfg: A.Resize(
                    height=cfg["height"],
                    width=cfg["width"],
                    interpolation=cfg.get("interpolation", 1),
                    p=cfg.get("p", 1.0),
                ),
            )
        )
        self.register(
            AugmentationSpec(
                "vertical_flip",
                lambda cfg: A.VerticalFlip(p=cfg.get("p", 0.5)),
            )
        )

    @staticmethod
    def _build_coarse_dropout(cfg: dict[str, Any]) -> A.BasicTransform:
        num_holes = cfg.get("num_holes", {"min": 1, "max": 4})
        min_holes = num_holes.get("min", 1)
        max_holes = num_holes.get("max", 4)
        max_height = cfg.get("max_height", 32)
        max_width = cfg.get("max_width", 32)
        p = cfg.get("p", 0.5)
        return A.CoarseDropout(
            num_holes_range=(min_holes, max_holes),
            hole_height_range=(1, max_height),
            hole_width_range=(1, max_width),
            p=p,
        )

    @staticmethod
    def _build_random_resized_crop(cfg: dict[str, Any]) -> A.BasicTransform:
        height = cfg["height"]
        width = cfg["width"]
        scale = _to_min_max(cfg.get("scale", {"min": 0.08, "max": 1.0}))
        ratio = _to_min_max(cfg.get("ratio", {"min": 0.75, "max": 1.3333333333}))
        interpolation = cfg.get("interpolation", 1)
        p = cfg.get("p", 1.0)

        try:
            return A.RandomResizedCrop(
                size=(height, width),
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
                p=p,
            )
        except Exception:
            return A.RandomResizedCrop(
                height=height,
                width=width,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
                p=p,
            )


@dataclass
class ImageAugmentationPipeline:
    albumentations_compose: A.Compose | None
    torchvision_postprocess: transforms.Compose

    def __call__(self, *args, **kwargs):
        if "image" in kwargs:
            image_input = kwargs["image"]
            return_dict = True
        elif len(args) == 1:
            image_input = args[0]
            return_dict = False
        else:
            raise TypeError("ImageAugmentationPipeline expects either a single image arg or keyword arg 'image'.")

        if isinstance(image_input, Image.Image):
            image_np = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            image_np = image_input
        else:
            raise TypeError("Unsupported image type. Expected PIL.Image.Image or numpy.ndarray.")

        if self.albumentations_compose is not None:
            image_np = self.albumentations_compose(image=image_np)["image"]
        image = Image.fromarray(image_np)
        output = self.torchvision_postprocess(image)
        if return_dict:
            return {"image": output}
        return output


def build_image_augmentations(
    config: dict[str, Any],
    resolution: int = None,
    center_crop: bool = None,
    random_flip: bool = None,
) -> ImageAugmentationPipeline:
    
    data_config = config.get("data", config)
    if resolution is None:
        resolution = data_config.get("resolution", 512)
    if center_crop is None:
        center_crop = data_config.get("center_crop", False)
    if random_flip is None:
        random_flip = data_config.get("random_flip", False)
        
    
    #augmentation_config = config.get("augmentation", {})# or {}
    augmentation_config = config["augmentation"]
    print(augmentation_config)

    registry = AugmentationRegistry()
    albumentations_transforms: list[A.BasicTransform] = []

    has_resize = "resize" in augmentation_config or "random_resized_crop" in augmentation_config
    has_hflip = "horizontal_flip" in augmentation_config

    # if not has_resize:
    #     albumentations_transforms.append(A.Resize(height=resolution, width=resolution, interpolation=1, p=1.0))
    #     if center_crop:
    #         albumentations_transforms.append(A.CenterCrop(height=resolution, width=resolution, p=1.0))
    #     else:
    #         albumentations_transforms.append(A.RandomCrop(height=resolution, width=resolution, p=1.0))

    # if random_flip and not has_hflip:
    #     albumentations_transforms.append(A.HorizontalFlip(p=0.5))

    for name, params in augmentation_config.items():
        transform = registry.build_transform(name, params)
        if transform is not None:
            albumentations_transforms.append(transform)

    albumentations_compose = A.Compose(albumentations_transforms) if albumentations_transforms else None

    spatial_augmentations = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ]

    torchvision_postprocess = transforms.Compose(
        spatial_augmentations +
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [ 0.5]),
        ]
    )

    return ImageAugmentationPipeline(
        albumentations_compose=albumentations_compose,
        torchvision_postprocess=torchvision_postprocess,
    )
