# ImageNet Baselines

Here, we use the following architectures:
* [VGGNet (Simonyan and Zisserman, 2015)](https://arxiv.org/abs/1409.1556) : VGG16,
* [ResNet (He et al., 2016)](https://arxiv.org/abs/1512.03385) : R18, R50.

## ImageNet

In summary,
* Use 1,281,167 train examples, and 50,000 valid examples from 1000 classes.
* Use train data augmentation consisting of random resized cropping and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 256, and base learning rate 0.1.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

### Train Models

Run the following command lines to train models with batch size 256 and base learning rate 0.1:
```
python scripts/train.py \
    --config_file ./configs/ImageNet1k_{NETWORK_NAME}_SGD.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 256 --learning_rate 0.1 --weight_decay 1e-4 \
    --seed 42 --output_dir ./outputs/ImageNet1k_{NETWORK_NAME}/SGD/s42_e100_b256_wd1e-4/
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/ImageNet1k_{NETWORK_NAME}_SGD.yaml \
    --weight_file ./outputs/ImageNet1k_{NETWORK_NAME}/SGD/s42_e100_b{BATCH_SIZE}_wd1e-4/best_acc1
```

### Results

| Network          | Batch Size | Base LR | Valid ACC / NLL / cNLL | Train Runtime        | Misc. |
| :-               | :-:        | :-:     | :-:                    | :-:                  | :-:   |
| VGG16-ReLU       | 256        | 0.01    | 72.32 / 1.114 / 1.104  | 15.8 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k/20220212202852.log) |
| VGG16-BN-ReLU    | 256        | 0.01    | 72.65 / 1.098 / 1.084  | 19.3 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k/20220213124035.log) |
| R18-BN-ReLU      | 256        | 0.1     | 70.45 / 1.210 / 1.204  | 5.1 hrs. (8 TPUv3)   | [log](./scripts/logs/ImageNet1k/20220212063003.log) |
|                  | 1024       | 0.4     | 70.53 / 1.184 / 1.183  | 4.2 hrs. (8 TPUv3)   | [log](./scripts/logs/ImageNet1k/20220214093712.log) |
| R50-BN-ReLU      | 256        | 0.1     | 76.78 / 0.927 / 0.913  | 8.9 hrs. (8 TPUv3)   | [log](./scripts/logs/ImageNet1k/20220212113453.log) |
|                  | 1024       | 0.4     | 76.86 / 0.909 / 0.903  | 7.0 hrs. (8 TPUv3)   | [log](./scripts/logs/ImageNet1k/20220214135046.log) |
| R101-BN-ReLU     | 256        | 0.1     | 78.72 / 0.845 / 0.823  | 13.1 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k/20220215155340.log) |
| R152-BN-ReLU     | 256        | 0.1     | 79.39 / 0.825 / 0.799  | 18.7 hrs. (8 TPUv3)  | [log](./scripts/logs/ImageNet1k/20220216162527.log) |
