# Model Training Notes

**Creation:** 2026-02-17 10:39:10

**Last Training session:** 2026-02-17 11:34:48

## Why was this model trained?

Iterative work getting VAE to work
## Configuration Summary

**Dataset Paths:**
- `/media/aris/Data/master2025dev/datasets/plastic`
- `/media/aris/Data/master2025dev/datasets/wood/all_wood_categories`

### Script Config

| Setting | Value |
|---------|-------|
| diffusion_parameters.lr_scheduler | _not set_ |
| diffusion_parameters.ddpm_num_steps | 1000 |
| diffusion_parameters.ddpm_num_inference_steps | 50 |
| diffusion_parameters.ddpm_beta_schedule | _not set_ |
| diffusion_parameters.use_ema | Yes |
| diffusion_parameters.lr_warmup_steps | 500 |
| diffusion_parameters._comment | ------------------ Secondary parameters: ----------------- |
| diffusion_parameters.adam_beta1 | 0.95 |
| diffusion_parameters.adam_beta2 | 0.999 |
| diffusion_parameters.adam_weight_decay | 1e-06 |
| diffusion_parameters.adam_epsilon | 1e-08 |
| diffusion_parameters.ema_inv_gamma | 1.0 |
| diffusion_parameters.ema_power | 0.75 |
| diffusion_parameters.ema_max_decay | 0.9999 |
| vae.use_vae | Yes |
| vae.vae_model_path | /media/aris/Data/master2025dev/aris_master/models/VAE/vae-ft-mse-840000-ema-pruned |
| hyperparameters.batch_size | 32 |
| hyperparameters.epochs | 1000 |
| hyperparameters.learning_rate | 0.00014 |
| hyperparameters.dataloader_num_workers | 4 |
| logging.logger | tensorboard |
| logging.output_dir | /media/aris/Data/master2025dev/aris_master/src/summarize_training/ |
| logging.comment | Iterative work getting VAE to work |
| logging.checkpointing_steps | 5 |
| logging.resume_from_checkpoint | Yes |
| logging.save_images_epochs | 10 |
| data.resolution | 128 |
| data.center_crop | Yes |
| data.random_flip | Yes |
| data.classes.impregnated_wood.data_dir | /media/aris/Data/master2025dev/datasets/wood/all_wood_categories |
| data.classes.impregnated_wood.sub_categories | impregnated_wood |
| data.classes.impregnated_wood.train_img_count | 1000 |
| data.classes.normal_wood.data_dir | /media/aris/Data/master2025dev/datasets/wood/all_wood_categories |
| data.classes.normal_wood.sub_categories | normal_wood |
| data.classes.normal_wood.train_img_count | 1000 |
| data.classes.soft_plastic.data_dir | /media/aris/Data/master2025dev/datasets/plastic |
| data.classes.soft_plastic.sub_categories | soft_plastic, hard_plastic |
| data.classes.soft_plastic.train_img_count | 1000 |
| model._class_name | UNet2DModel |
| model._diffusers_version | 0.36.0 |
| model.act_fn | silu |
| model.add_attention | Yes |
| model.attention_head_dim | 8 |
| model.attn_norm_num_groups | _not set_ |
| model.block_out_channels | 4 items |
| model.center_input_sample | No |
| model.class_embed_type | _not set_ |
| model.down_block_types | 4 items |
| model.downsample_padding | 1 |
| model.downsample_type | conv |
| model.dropout | 0.0 |
| model.flip_sin_to_cos | Yes |
| model.freq_shift | 0 |
| model.in_channels | 4 |
| model.layers_per_block | 2 |
| model.mid_block_scale_factor | 1 |
| model.mid_block_type | UNetMidBlock2D |
| model.norm_eps | 1e-05 |
| model.norm_num_groups | 32 |
| model.num_class_embeds | _not set_ |
| model.num_train_timesteps | _not set_ |
| model.out_channels | 4 |
| model.resnet_time_scale_shift | default |
| model.sample_size | 16 |
| model.time_embedding_dim | _not set_ |
| model.time_embedding_type | positional |
| model.up_block_types | 4 items |
| model.upsample_type | conv |

### Split Distribution

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 12,940 | 90.3% |
| Val | 1,390 | 9.7% |
| **Total** | **14,330** | **100.0%** |

### Category Distribution

#### Train Set

| Category | Images | Percentage |
|----------|--------|------------|
| soft_plastic | 6,474 | 50.0% |
| normal_wood | 4,952 | 38.3% |
| impregnated_wood | 1,514 | 11.7% |
| **Total** | **12,940** | **100.0%** |

#### Val Set

| Category | Images | Percentage |
|----------|--------|------------|
| soft_plastic | 672 | 48.3% |
| normal_wood | 550 | 39.6% |
| impregnated_wood | 168 | 12.1% |
| **Total** | **1,390** | **100.0%** |

## Sampling Results

## Conclusion
