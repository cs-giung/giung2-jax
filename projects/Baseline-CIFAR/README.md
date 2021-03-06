# CIFAR Baselines

Here, we use the following architectures:
* [VGGNet (Simonyan and Zisserman, 2015)](https://arxiv.org/abs/1409.1556) : VGG11, VGG13, VGG16, VGG19,
* [ResNet (He et al., 2016)](https://arxiv.org/abs/1512.03385) : R20, R32, R44, R56,
* [WideResNet (Zagoruyko and Komodakis, 2016)](https://arxiv.org/abs/1605.07146) : WRN28x1, WRN28x5, WRN28x10.

Note that the original VGG architecture does not consider CIFAR datasets.
Here, we have two modifications for our experiments using VGG architectures:
1. We omit the last MLP structure,
    ```python
    MODEL.BACKBONE.VGGNET.MLP_HIDDENS = [] # [4096, 4096,]
    ```

2. We also test VGG architectures with [Batch Normalization (Ioffe and Szegedy, 2015)](https://arxiv.org/abs/1502.03167) layers,
    ```python
    MODEL.BACKBONE.VGGNET.NORM_LAYERS = "BatchNorm2d" # "NONE"
    ```

## Command Lines

In summary,
* Use 45,000 train examples, 5,000 valid examples, and 10,000 test examples from 10/100 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.1.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

### Train Models

Run the following command lines to train models:
```
python scripts/train.py \
    --config_file ./configs/{DATASET_NAME}_{NETWORK_NAME}.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/{DATASET_NAME}_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/
```

### Evaluate Models

Run the following command lines to evaluate models:
```
python scripts/eval.py \
    --config_file ./configs/{DATASET_NAME}_{NETWORK_NAME}.yaml \
    --weight_file ./outputs/{DATASET_NAME}_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/best_acc1
```

## CIFAR-10

| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  | :-:   |
| VGG11-ReLU       | 99.96 / 0.005 / 0.036  | 91.78 / 0.347 / 0.262  | 90.98 / 0.375 / 0.281  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320100424.log) |
| VGG13-ReLU       | 99.95 / 0.004 / 0.028  | 92.92 / 0.295 / 0.220  | 92.55 / 0.316 / 0.232  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320100425.log) |
| VGG16-ReLU       | 99.96 / 0.003 / 0.026  | 92.42 / 0.316 / 0.237  | 92.25 / 0.340 / 0.252  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320100426.log) |
| VGG19-ReLU       | -                      | -                      | -                      | -                    | - |
| VGG11-BN-ReLU    | 100.0 / 0.001 / 0.017  | 92.60 / 0.302 / 0.247  | 92.25 / 0.311 / 0.257  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320102408.log) |
| VGG13-BN-ReLU    | 100.0 / 0.001 / 0.013  | 94.70 / 0.225 / 0.189  | 94.04 / 0.243 / 0.205  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320102858.log) |
| VGG16-BN-ReLU    | 100.0 / 0.001 / 0.020  | 94.30 / 0.272 / 0.215  | 93.93 / 0.282 / 0.223  | 0.3 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320103354.log) |
| VGG19-BN-ReLU    | 100.0 / 0.001 / 0.022  | 94.30 / 0.295 / 0.227  | 93.82 / 0.305 / 0.233  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220320103443.log) |
| R20-BN-ReLU      | 99.91 / 0.008 / 0.029  | 92.90 / 0.255 / 0.218  | 92.49 / 0.277 / 0.235  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205025405.log) |
| R32-BN-ReLU      | 99.98 / 0.002 / 0.018  | 93.98 / 0.251 / 0.201  | 93.43 / 0.273 / 0.218  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205024428.log) |
| R32x2-BN-ReLU    | 100.0 / 0.000 / 0.012  | 95.00 / 0.220 / 0.173  | 94.86 / 0.210 / 0.171  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220222144259.log) |
| R32x3-BN-ReLU    | 100.0 / 0.000 / 0.009  | 95.34 / 0.194 / 0.160  | 95.41 / 0.188 / 0.157  | 0.7 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220222154258.log) |
| R32x4-BN-ReLU    | 100.0 / 0.000 / 0.008  | 95.98 / 0.171 / 0.145  | 95.83 / 0.168 / 0.144  | 0.9 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220222105140.log) |
| R44-BN-ReLU      | 99.98 / 0.001 / 0.018  | 94.40 / 0.260 / 0.194  | 93.85 / 0.261 / 0.199  | 0.3 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205030945.log) |
| R56-BN-ReLU      | 99.99 / 0.001 / 0.016  | 94.26 / 0.264 / 0.194  | 93.75 / 0.279 / 0.206  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205032630.log) |
| WRN16x8-BN-ReLU  | 100.0 / 0.001 / 0.003  | 95.48 / 0.148 / 0.141  | 95.67 / 0.151 / 0.145  | 0.9 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220216124110.log) |
| WRN22x2-BN-ReLU  | 100.0 / 0.001 / 0.010  | 94.84 / 0.201 / 0.170  | 94.49 / 0.205 / 0.175  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220218072134.log) |
| WRN28x1-BN-ReLU  | 99.97 / 0.003 / 0.022  | 93.22 / 0.263 / 0.216  | 92.87 / 0.267 / 0.221  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205024053.log) |
| WRN28x5-BN-ReLU  | 100.0 / 0.000 / 0.008  | 96.04 / 0.173 / 0.147  | 95.75 / 0.166 / 0.143  | 1.0 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205082640.log) |
|                  | 100.0 / 0.000 / 0.007  | 96.10 / 0.154 / 0.133  | 95.79 / 0.167 / 0.144  | 0.8 hrs. (2 RTX3090) | [log](./scripts/logs/C10/20220205120405.log) |
|                  | 100.0 / 0.000 / 0.009  | 95.80 / 0.175 / 0.147  | 95.67 / 0.170 / 0.145  | 0.6 hrs. (4 RTX3090) | [log](./scripts/logs/C10/20220205120400.log) |
| WRN28x10-BN-ReLU | 100.0 / 0.000 / 0.008  | 96.24 / 0.164 / 0.141  | 96.16 / 0.157 / 0.137  | 2.7 hrs. (1 RTX3090) | [log](./scripts/logs/C10/20220205121207.log) |
|                  | 100.0 / 0.001 / 0.008  | 96.42 / 0.154 / 0.133  | 96.09 / 0.157 / 0.137  | 2.1 hrs. (2 RTX3090) | [log](./scripts/logs/C10/20220205125410.log) |
|                  | 100.0 / 0.001 / 0.008  | 96.38 / 0.157 / 0.134  | 96.30 / 0.153 / 0.132  | 1.8 hrs. (4 RTX3090) | [log](./scripts/logs/C10/20220205123923.log) |
|                  | 100.0 / 0.000 / 0.010  | 96.18 / 0.177 / 0.146  | 96.08 / 0.170 / 0.142  | 0.6 hrs. (8 TPUv3)   | [log](./scripts/logs/C10/20220206051109.log) |

## CIFAR-100

| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  | :-:   |
| VGG11-ReLU       | 99.77 / 0.017 / 0.179  | 62.12 / 2.111 / 1.473  | 63.57 / 2.059 / 1.442  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320101322.log) |
| VGG13-ReLU       | 99.76 / 0.015 / 0.148  | 66.54 / 1.890 / 1.300  | 66.66 / 1.848 / 1.281  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320101648.log) |
| VGG16-ReLU       | 99.75 / 0.015 / 0.152  | 69.18 / 1.786 / 1.153  | 69.84 / 1.811 / 1.163  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320101939.log) |
| VGG19-ReLU       | 99.76 / 0.010 / 0.157  | 69.36 / 2.221 / 1.206  | 69.73 / 2.197 / 1.199  | 0.3 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320102220.log) |
| VGG11-BN-ReLU    | 99.98 / 0.005 / 0.010  | 70.74 / 1.199 / 1.187  | 70.57 / 1.186 / 1.176  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320104214.log) |
| VGG13-BN-ReLU    | 99.98 / 0.004 / 0.009  | 73.96 / 1.046 / 1.035  | 74.46 / 1.013 / 1.007  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320104224.log) |
| VGG16-BN-ReLU    | 99.98 / 0.002 / 0.028  | 74.06 / 1.211 / 1.064  | 74.59 / 1.184 / 1.050  | 0.3 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320105044.log) |
| VGG19-BN-ReLU    | 99.97 / 0.003 / 0.075  | 73.46 / 1.422 / 1.133  | 73.68 / 1.385 / 1.115  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220320105158.log) |
| R20-BN-ReLU      | 92.43 / 0.271 / 0.383  | 68.08 / 1.220 / 1.129  | 68.19 / 1.228 / 1.138  | 0.1 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205025405.log) |
| R32-BN-ReLU      | 98.16 / 0.089 / 0.215  | 70.46 / 1.256 / 1.085  | 70.62 / 1.232 / 1.075  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205024428.log) |
| R32x2-BN-ReLU    | 99.98 / 0.005 / 0.033  | 74.66 / 1.095 / 0.985  | 75.04 / 1.076 / 0.970  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220222144259.log) |
| R32x3-BN-ReLU    | 99.98 / 0.002 / 0.010  | 77.06 / 0.937 / 0.898  | 76.67 / 0.951 / 0.911  | 0.7 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220222154258.log) |
| R32x4-BN-ReLU    | 99.98 / 0.002 / 0.005  | 77.98 / 0.888 / 0.872  | 77.78 / 0.884 / 0.869  | 0.9 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220222105140.log) |
| R44-BN-ReLU      | 99.51 / 0.038 / 0.141  | 71.02 / 1.270 / 1.064  | 70.74 / 1.262 / 1.059  | 0.3 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205030945.log) |
| R56-BN-ReLU      | 99.72 / 0.022 / 0.116  | 71.34 / 1.302 / 1.055  | 71.80 / 1.278 / 1.043  | 0.4 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205032630.log) |
| WRN16x8-BN-ReLU  | 99.99 / 0.002 / 0.001  | 78.46 / 0.885 / 0.880  | 78.65 / 0.893 / 0.887  | 0.9 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220216124110.log) |
| WRN22x2-BN-ReLU  | 99.95 / 0.011 / 0.060  | 72.94 / 1.168 / 1.035  | 73.57 / 1.125 / 1.007  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220218072134.log) |
| WRN28x1-BN-ReLU  | 96.43 / 0.149 / 0.269  | 69.44 / 1.232 / 1.106  | 68.86 / 1.262 / 1.131  | 0.2 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205024053.log) |
| WRN28x5-BN-ReLU  | 99.99 / 0.002 / 0.003  | 78.62 / 0.880 / 0.875  | 78.55 / 0.873 / 0.869  | 1.0 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205082640.log) |
|                  | 99.98 / 0.001 / 0.004  | 79.16 / 0.866 / 0.852  | 78.66 / 0.862 / 0.850  | 0.7 hrs. (2 RTX3090) | [log](./scripts/logs/C100/20220205120424.log) |
|                  | 99.99 / 0.001 / 0.004  | 79.32 / 0.897 / 0.865  | 78.80 / 0.865 / 0.842  | 0.6 hrs. (4 RTX3090) | [log](./scripts/logs/C100/20220205120345.log) |
| WRN28x10-BN-ReLU | 99.99 / 0.002 / 0.002  | 80.74 / 0.807 / 0.807  | 80.33 / 0.817 / 0.817  | 2.7 hrs. (1 RTX3090) | [log](./scripts/logs/C100/20220205121149.log) |
|                  | 99.99 / 0.001 / 0.003  | 80.42 / 0.821 / 0.816  | 80.38 / 0.805 / 0.803  | 1.8 hrs. (2 RTX3090) | [log](./scripts/logs/C100/20220205124541.log) |
|                  | 99.98 / 0.001 / 0.003  | 80.22 / 0.844 / 0.826  | 80.49 / 0.810 / 0.798  | 1.6 hrs. (4 RTX3090) | [log](./scripts/logs/C100/20220205123726.log) |
|                  | 99.98 / 0.001 / 0.004  | 80.04 / 0.900 / 0.844  | 80.63 / 0.843 / 0.802  | 0.6 hrs. (8 TPUv3)   | [log](./scripts/logs/C100/20220206054931.log) |
