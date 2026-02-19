# Model Training Notes

**Creation:** 2026-02-17 15:35:52

**Last Training session:** 2026-02-17 15:35:52

## Why was this model trained?

First resnet training attempt. Training on 1000 real images, no synthetic data. Let's see how it goes.

## Configuration Summary

**Dataset Path:** `/mnt/master2025dev/datasets/wood/all_wood_categories`

### Script Config

| Setting | Value |
|---------|-------|
| hyperparameters.batch_size | 16 |
| hyperparameters.epochs | 100 |
| hyperparameters.learning_rate | 0.0001 |
| hyperparameters.dataloader_num_workers | 4 |
| logging.logger | tensorboard |
| logging.output_dir | /home/ap/cloud/Master/aris_master/testing/02-17_1000_real_images_no_synthetic |
| logging.comment | First resnet training attempt. Training on 1000 real images, no synthetic data. Let's see how it goes. |
| logging.checkpointing_steps | 5 |
| logging.resume_from_checkpoint | Yes |
| logging.checkpoint_dir | _not set_ |
| logging.seed | 42 |
| data.resolution | 128 |
| data.center_crop | Yes |
| data.random_flip | Yes |
| data.real_image_count | 1000 |
| data.synthetic_image_count | _not set_ |
| data.synthetic_data_dir | _not set_ |
| data.classes.impregnated_wood.data_dir | /mnt/master2025dev/datasets/wood/all_wood_categories/ |
| data.classes.impregnated_wood.sub_categories | 4 items |
| data.classes.normal_wood.data_dir | /mnt/master2025dev/datasets/wood/all_wood_categories/ |
| data.classes.normal_wood.sub_categories | normal_wood, normal_wood_painted |

### Split Distribution

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 1,000 | 48.2% |
| Val | 1,075 | 51.8% |
| **Total** | **2,075** | **100.0%** |

### Category Distribution

#### Train Set

| Category | Images | Percentage |
|----------|--------|------------|
| impregnated_wood | 500 | 50.0% |
| normal_wood | 500 | 50.0% |
| **Total** | **1,000** | **100.0%** |

#### Val Set

| Category | Images | Percentage |
|----------|--------|------------|
| normal_wood | 635 | 59.1% |
| impregnated_wood | 440 | 40.9% |
| **Total** | **1,075** | **100.0%** |

