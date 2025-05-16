# Run commands for Adaptive Number of Participants

⚠️ Important: Follow the Prerequisites steps to set up enviroment.

⚠ Script argument: all scripts have a `--device_id`, which defines the GPU idx and `--dynamic_clients` that runs ISP technique.

## CIFAR-10 Experiments

```bash
python scripts/cifar10_script.py > cifar10_log_script.txt &
```

`--scheduler_clients` runs AdaFL strategy.

## Gradient Compression Experiments

```bash
python scripts/grad_compression_script.py > grad_compression_log_script.txt &
```

## ImageNet Experiments

```bash
python scripts/imagenet_script.py > imagenet_log_script.txt &
```