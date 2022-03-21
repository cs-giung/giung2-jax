# Downsampled ImageNet Baselines

Here, we use the following architectures:
* [VGGNet (Simonyan and Zisserman, 2015)](https://arxiv.org/abs/1409.1556) : VGG11, VGG13, VGG16, VGG19,
* [ResNet (He et al., 2016)](https://arxiv.org/abs/1512.03385) : R18, R34, R50, R101, R152,
* [WideResNet (Zagoruyko and Komodakis, 2016)](https://arxiv.org/abs/1605.07146) : WRN28x10.

Note that the original architectures do not consider downsampled variants of the ImageNet dataset.
Here, we have some modifications for our experiments on downsampled datasets using those architectures:
1. We reduce the number of channels in the last FC layers from 4,096 to 512 for VGGNet,
    ```python
    MODEL.BACKBONE.VGGNET.MLP_HIDDENS = [512, 512,] # [4096, 4096,]
    ```

2. We also test VGG architectures with [Batch Normalization (Ioffe and Szegedy, 2015)](https://arxiv.org/abs/1502.03167) layers,
    ```python
    MODEL.BACKBONE.VGGNET.NORM_LAYERS = "BatchNorm2d" # "NONE"
    ```

3. We remove the pooling layer and modify the convolution layer in the first block of ResNet,
    ```python
    MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_POOL_LAYER = False # True
    MODEL.BACKBONE.RESNET.FIRST_BLOCK.CONV_KSP = [3, 1, 1,] # [7, 2, 3,]
    ```

## TinyImageNet-200

In summary,
* Use 90,000 train examples, 10,000 valid examples, and 10,000 test examples from 200 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.1.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

### Train Models

Run the following command lines to train models:
```
python scripts/train.py \
    --config_file ./configs/TIN200_{NETWORK_NAME}.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/TIN200_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/TIN200_{NETWORK_NAME}.yaml \
    --weight_file ./outputs/TIN200_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/best_acc1
```

### Results

| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  | :-:   |
| VGG11-ReLU       | 99.87 / 0.008 / 0.249  | 52.48 / 3.825 / 2.084  | 51.81 / 3.936 / 2.118  | 0.4 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207171004.log) |
| VGG11-BN-ReLU    | 99.96 / 0.006 / 0.129  | 56.50 / 2.492 / 1.816  | 55.85 / 2.531 / 1.845  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207173628.log) |
| VGG13-ReLU       | 99.92 / 0.005 / 0.228  | 53.43 / 4.107 / 2.020  | 53.28 / 4.156 / 2.042  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207180343.log) |
| VGG13-BN-ReLU    | 99.97 / 0.006 / 0.114  | 58.44 / 2.316 / 1.707  | 58.23 / 2.373 / 1.736  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207183140.log) |
| VGG16-ReLU       | 99.90 / 0.005 / 0.235  | 53.48 / 4.254 / 2.027  | 53.83 / 4.274 / 2.032  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207190047.log) |
| VGG16-BN-ReLU    | 99.96 / 0.004 / 0.104  | 58.74 / 2.435 / 1.719  | 57.76 / 2.503 / 1.760  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207193035.log) |
| VGG19-ReLU       | 99.87 / 0.006 / 0.290  | 52.43 / 4.798 / 2.061  | 52.28 / 4.914 / 2.080  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207200147.log) |
| VGG19-BN-ReLU    | 99.94 / 0.005 / 0.117  | 58.77 / 2.534 / 1.750  | 57.38 / 2.639 / 1.806  | 0.5 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220207203327.log) |
| R18-BN-ReLU      | 99.98 / 0.005 / 0.022  | 66.07 / 1.544 / 1.459  | 65.52 / 1.581 / 1.486  | 0.6 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220206063249.log) |
|                  | 99.98 / 0.009 / 0.017  | 64.98 / 1.497 / 1.485  | 64.98 / 1.533 / 1.519  | 1.6 hrs. (1 RTX3090) | [log](./scripts/logs/TIN200/20220206151326.log) |
|                  | 99.98 / 0.007 / 0.018  | 65.84 / 1.488 / 1.464  | 65.44 / 1.529 / 1.499  | 1.1 hrs. (2 RTX3090) | [log](./scripts/logs/TIN200/20220205132411.log) |
|                  | 99.98 / 0.006 / 0.018  | 66.48 / 1.480 / 1.439  | 65.51 / 1.547 / 1.495  | 0.8 hrs. (4 RTX3090) | [log](./scripts/logs/TIN200/20220205121009.log) |
| R34-BN-ReLU      | 99.97 / 0.002 / 0.010  | 67.58 / 1.500 / 1.416  | 66.88 / 1.558 / 1.462  | 0.8 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220224152246.log) |
| R50-BN-ReLU      | 99.97 / 0.004 / 0.016  | 69.80 / 1.343 / 1.265  | 69.36 / 1.378 / 1.296  | 1.1 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220206071326.log) |
|                  | 99.98 / 0.005 / 0.008  | 69.73 / 1.305 / 1.300  | 69.20 / 1.326 / 1.321  | 5.7 hrs. (1 RTX3090) | [log](./scripts/logs/TIN200/20220206151219.log) |
|                  | 99.98 / 0.004 / 0.009  | 69.53 / 1.299 / 1.288  | 69.16 / 1.342 / 1.327  | 3.5 hrs. (2 RTX3090) | [log](./scripts/logs/TIN200/20220205132437.log) |
|                  | 99.98 / 0.003 / 0.010  | 70.10 / 1.309 / 1.276  | 69.22 / 1.353 / 1.315  | 2.2 hrs. (4 RTX3090) | [log](./scripts/logs/TIN200/20220205120813.log) |
| R101-BN-ReLU     | 99.97 / 0.003 / 0.017  | 71.18 / 1.321 / 1.218  | 70.81 / 1.362 / 1.251  | 1.6 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220224130549.log) |
| R152-BN-ReLU     | 99.98 / 0.002 / 0.014  | 71.66 / 1.294 / 1.197  | 70.81 / 1.346 / 1.231  | 2.0 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220224104514.log) |

## ImageNet-1k_x32

In summary,
* Use 1,281,167 train examples, and 50,000 valid examples from 1000 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.01.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

### Train Models

Run the following command lines to train models:
```
python scripts/train.py \
    --config_file ./configs/ImageNet1k_x32_{NETWORK_NAME}.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.01 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/ImageNet1k_x32_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/ImageNet1k_x32_{NETWORK_NAME}.yaml \
    --weight_file ./outputs/ImageNet1k_x32_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/best_acc1
```

### Results

| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                  | :-:   |
| R18-BN-ReLU      | 69.10 / 1.225 / 1.253  | 55.40 / 1.931 / 1.917  |  3.3 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x32/20220320074932.log) |
| R34-BN-ReLU      | 78.50 / 0.807 / 0.861  | 59.28 / 1.783 / 1.742  |  4.5 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x32/20220320110925.log) |
| R50-BN-ReLU      | 83.11 / 0.614 / 0.675  | 62.45 / 1.682 / 1.616  |  6.0 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x32/20220320153708.log) |
| WRN28x10-BN-ReLU | 90.64 / 0.333 / 0.412  | 60.86 / 1.875 / 1.775  | 14.8 hrs. (8 TPUv2)  | [log](./scripts/logs/ImageNet1k_x32/20220317191255.log) |

## ImageNet-1k_x64

In summary,
* Use 1,281,167 train examples, and 50,000 valid examples from 1000 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.01.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

### Train Models

Run the following command lines to train models:
```
python scripts/train.py \
    --config_file ./configs/ImageNet1k_x64_{NETWORK_NAME}.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.01 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/ImageNet1k_x64_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/ImageNet1k_x64_{NETWORK_NAME}.yaml \
    --weight_file ./outputs/ImageNet1k_x64_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/best_acc1
```

### Results

| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                  | :-:   |
| R18-BN-ReLU      | 79.11 / 0.788 / 0.810  | 65.63 / 1.420 / 1.412  |  8.5 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x64/20220317134634.log) |
| R34-BN-ReLU      | 89.55 / 0.368 / 0.419  | 68.85 / 1.332 / 1.286  | 10.9 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x64/20220317221604.log) |
| R50-BN-ReLU      | 92.14 / 0.271 / 0.315  | 71.92 / 1.220 / 1.168  | 15.7 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x64/20220318091308.log) |
| R101-BN-ReLU     | 94.38 / 0.188 / 0.240  | 72.46 / 1.237 / 1.158  | 21.6 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x64/20220319005752.log) |
| R152-BN-ReLU     | 95.51 / 0.147 / 0.201  | 72.96 / 1.253 / 1.154  | 28.2 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k_x64/20220318192432.log) |
