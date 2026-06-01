# Data augmentations

## affine augmentation

![affine_augmentation_samples.png](aug_images/affine_augmentation_samples.png)

## clahe augmentation

![clahe_augmentation_samples.png](aug_images/clahe_augmentation_samples.png)

## brightness_contrast augmentation

![brightness_contrast_augmentation_samples.png](aug_images/brightness_contrast_augmentation_samples.png)

## coarse_dropout augmentation

![coarse_dropout_augmentation_samples.png](aug_images/coarse_dropout_augmentation_samples.png)

## color_jitter augmentation

![color_jitter_augmentation_samples.png](aug_images/color_jitter_augmentation_samples.png)

## defocus augmentation

![defocus_augmentation_samples.png](aug_images/defocus_augmentation_samples.png)

## gauss_noise augmentation

![gauss_noise_augmentation_samples.png](aug_images/gauss_noise_augmentation_samples.png)

## horizontal_flip augmentation

![horizontal_flip_augmentation_samples.png](aug_images/horizontal_flip_augmentation_samples.png)

## iso_noise augmentation

![iso_noise_augmentation_samples.png](aug_images/iso_noise_augmentation_samples.png)

## random_resized_crop augmentation

![random_resized_crop_augmentation_samples.png](aug_images/random_resized_crop_augmentation_samples.png)

## random_rotate_90 augmentation

![random_rotate_90_augmentation_samples.png](aug_images/random_rotate_90_augmentation_samples.png)

## random_shadow augmentation

![random_shadow_augmentation_samples.png](aug_images/random_shadow_augmentation_samples.png)

## reszie augmentation

![reszie_augmentation_samples.png](aug_images/reszie_augmentation_samples.png)

## vertical_flip augmentation

![vertical_flip_augmentation_samples.png](aug_images/vertical_flip_augmentation_samples.png)

## combined augmentation

![combined_augmentation_samples.png](aug_images/combined_augmentation_samples.png)

Augmentations:

```json
  "augmentation": {
    "affine": {
      "interpolation": 2,
      "p": 0.0,
      "rotate": {
        "max": 45,
        "min": -45
      },
      "scale": {
        "max": 1.2,
        "min": 0.8
      },
      "shear": {
        "max": 10,
        "min": -10
      },
      "translate_percent": {
        "max": 0.15,
        "min": -0.15
      }
    },
    "brightness_contrast": {
      "brightness_limit": {
        "max": 0.4,
        "min": -0.2
      },
      "contrast_limit": {
        "max": 0.2,
        "min": -0.1
      },
      "p": 0.2
    },
    "clahe": {
      "clip_limit": 2.0,
      "p": 0.25
    },
    "coarse_dropout": {
      "max_height": 100,
      "max_width": 100,
      "num_holes": {
        "max": 8,
        "min": 2
      },
      "p": 0.2
    },
    "color_jitter": {
      "brightness": {
        "max": 1.2,
        "min": 0.8
      },
      "contrast": {
        "max": 1.2,
        "min": 0.8
      },
      "hue": 0.2,
      "p": 0.2,
      "saturation": {
        "max": 1.2,
        "min": 0.8
      }
    },
    "defocus": {
      "alias_blur": {
        "max": 2,
        "min": 1.5
      },
      "p": 0.3,
      "radius": {
        "max": 13,
        "min": 5
      }
    },
    "gauss_noise": {
      "mean_range": {
        "max": 1,
        "min": -1
      },
      "p": 0.0,
      "std_range": {
        "max": 10,
        "min": 9
      }
    },
    "horizontal_flip": {
      "p": 0.4
    },
    "iso_noise": {
      "color_shift": {
        "max": 0.6,
        "min": 0.2
      },
      "intensity": {
        "max": 0.7,
        "min": 0.2
      },
      "p": 0.5
    },
    "random_resized_crop": {
      "height": 400,
      "p": 0.0,
      "ratio": {
        "max": 1.1,
        "min": 0.9
      },
      "scale": {
        "max": 1.0,
        "min": 0.08
      },
      "width": 600
    },
    "random_rotate_90": {
      "p": 0.0
    },
    "random_shadow": {
      "num_shadows_limit": {
        "max": 2,
        "min": 1
      },
      "p": 0.2,
      "shadow_dimension": 5
    },
    "resize": {
      "height": 1200,
      "interpolation": 0,
      "p": 0.0,
      "width": 1200
    },
    "vertical_flip": {
      "p": 0.0
    }
  }
```
