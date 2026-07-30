# Optional depth auxiliary training

## Overview

The original classification + Fourier reconstruction training path remains the default.
Depth supervision is enabled only when `conf.depth_aux_enabled=True` or when
`train.py` is launched with `--enable_depth_aux`.

The shared MiniFASNet feature map feeds three training heads:

1. classification (`CrossEntropyLoss`);
2. Fourier reconstruction (`MSELoss`, unchanged);
3. a 40x40 depth decoder supervised by the prepared full-image Depth Anything V2 maps.

The depth decoder is training-only and is not returned in evaluation mode, so the
deployment classification path remains unchanged.

## Configuration

The default paths are configured in `src/default_config.py`:

- `depth_root_path`: CelebA_Spoof full-image Depth Anything V2 depth root;
- `additional_depth_root_paths`: CVPR23-FAS-WILD, DISFA and FF++ full-image Depth Anything V2 roots;
- `depth_target_mode`: `class_conditioned` uses the prepared depth map for class 1 and a zero
  target for attack classes;
- `depth_loss_weight`: final depth loss weight (default `0.1`);
- `depth_gradient_weight`: local gradient loss weight (default `0.1`);
- `depth_loss_warmup_epochs`: linear warm-up duration (default `5`).

The loader skips samples without a prepared depth file. RGB and depth use the same
random crop, rotation and flip; color jitter is applied only to RGB. The 80x80
stored depth image is transformed to the configured 40x40 supervision target.

## Loss

```text
L = 0.5 L_cls + 0.5 L_fourier + lambda_depth(t) L_depth
L_depth = smooth_l1 + depth_gradient_weight * gradient_loss
```

Both terms are averaged over the complete depth map. No face mask or valid-face
mask is applied, so background depth contributes to the supervision as well.

## Run

```bash
conda activate SilentFAS
python train.py --device_ids 0 --patch_info 2.7_80x80 --enable_depth_aux
```

The TXT log is written below:

```text
/userdata/lwk/FaceFakeDetection/SpoofDetection/Silent-Face-Anti-Spoofing-master/saved_logs/train_logs
```

## Compatibility

Without `--enable_depth_aux` and with `conf.depth_aux_enabled=False`, the dataset
returns `(image, fourier_target, label)`, the model returns its original outputs,
and the original two-loss training path is retained.
