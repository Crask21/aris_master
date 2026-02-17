# Model Training Notes

**Creation:** 2026-02-17 12:37:19

**Last Training session:** 2026-02-17 14:11:28

## Why was this model trained?

Bla. bla. bla.
## Configuration Summary

**Dataset Path:** `/media/aris/Data/master2025dev/datasets/wood/all_wood_categories`

### Script Config

| Setting | Value |
|---------|-------|
| hyperparameters.batch_size | 16 |
| hyperparameters.epochs | 5 |
| hyperparameters.learning_rate | 0.0001 |
| hyperparameters.dataloader_num_workers | 4 |
| logging.logger | tensorboard |
| logging.output_dir | testing/testing_dir |
| logging.comment | Bla. bla. bla. |
| logging.checkpointing_steps | 1 |
| logging.resume_from_checkpoint | Yes |
| logging.checkpoint_dir | testing/testing_dir/resnet18/resnet18_epoch2_valacc55.96_val_loss0.6696.ckpt |
| data.resolution | 128 |
| data.center_crop | Yes |
| data.random_flip | Yes |
| data.real_image_count | 200 |
| data.synthetic_image_count | _not set_ |
| data.synthetic_data_dir | _not set_ |
| data.classes.impregnated_wood.data_dir | /media/aris/Data/master2025dev/datasets/wood/all_wood_categories |
| data.classes.impregnated_wood.sub_categories | 4 items |
| data.classes.normal_wood.data_dir | /media/aris/Data/master2025dev/datasets/wood/all_wood_categories |
| data.classes.normal_wood.sub_categories | normal_wood |

### Split Distribution

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 200 | 16.8% |
| Val | 990 | 83.2% |
| **Total** | **1,190** | **100.0%** |

### Category Distribution

#### Train Set

| Category | Images | Percentage |
|----------|--------|------------|
| impregnated_wood | 100 | 50.0% |
| normal_wood | 100 | 50.0% |
| **Total** | **200** | **100.0%** |

#### Val Set

| Category | Images | Percentage |
|----------|--------|------------|
| normal_wood | 550 | 55.6% |
| impregnated_wood | 440 | 44.4% |
| **Total** | **990** | **100.0%** |

## Sampling Results

## Conclusion
