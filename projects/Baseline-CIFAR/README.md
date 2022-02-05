# CIFAR Baselines

In summary,
* Use 45,000 train examples, 5,000 valid examples, and 10,000 test examples from 10/100 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.1.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

## CIFAR-10

Run the following command lines:
```
python scripts/train.py \
    --config_file ./configs/C10_{NETWORK_NAME}_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C10_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/
```
```
python scripts/eval.py \
    --config_file ./configs/C10_{NETWORK_NAME}_SGD.yaml \
    --weight_file ./outputs/C10_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/
```

As a result, we get the following:
| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  |
| R20-BN-ReLU      | 99.91 / 0.008 / 0.029  | 92.90 / 0.255 / 0.218  | 92.49 / 0.277 / 0.235  | 0.1 hrs. (1 RTX3090) |  
| R32-BN-ReLU      | 99.98 / 0.002 / 0.018  | 93.98 / 0.251 / 0.201  | 93.43 / 0.273 / 0.218  | 0.2 hrs. (1 RTX3090) |
| R44-BN-ReLU      | 99.98 / 0.001 / 0.018  | 94.40 / 0.260 / 0.194  | 93.85 / 0.261 / 0.199  | 0.3 hrs. (1 RTX3090) |
| R56-BN-ReLU      | 99.99 / 0.001 / 0.016  | 94.26 / 0.264 / 0.194  | 93.75 / 0.279 / 0.206  | 0.4 hrs. (1 RTX3090) |
| WRN28x1-BN-ReLU  | 99.97 / 0.003 / 0.022  | 93.22 / 0.263 / 0.216  | 92.87 / 0.267 / 0.221  | 0.2 hrs. (1 RTX3090) |
| WRN28x5-BN-ReLU  | 100.0 / 0.000 / 0.007  | 95.78 / 0.165 / 0.140  | 95.77 / 0.176 / 0.149  | 0.7 hrs. (2 RTX3090) |
| WRN28x10-BN-ReLU | 100.0 / 0.000 / 0.008  | 96.38 / 0.155 / 0.133  | 96.17 / 0.152 / 0.133  | 2.1 hrs. (2 RTX3090) |
|                  | 100.0 / 0.000 / 0.008  | 96.04 / 0.155 / 0.134  | 96.16 / 0.158 / 0.137  | 1.7 hrs. (4 RTX3090) |

## CIFAR-100

Run the following command lines:
```
python scripts/train.py \
    --config_file ./configs/C100_{NETWORK_NAME}_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C100_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/
```
```
python scripts/eval.py \
    --config_file ./configs/C100_{NETWORK_NAME}_SGD.yaml \
    --weight_file ./outputs/C100_{NETWORK_NAME}/SGD/s42_e200_wd5e-4/
```

As a result, we get the following:
| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  |
| R20-BN-ReLU      | 92.43 / 0.271 / 0.383  | 68.10 / 1.220 / 1.129  | 68.19 / 1.228 / 1.138  | 0.1 hrs. (1 RTX3090) |
| R32-BN-ReLU      | 98.16 / 0.089 / 0.215  | 70.46 / 1.256 / 1.085  | 70.62 / 1.232 / 1.075  | 0.2 hrs. (1 RTX3090) |
| R44-BN-ReLU      | 99.51 / 0.038 / 0.141  | 71.02 / 1.270 / 1.064  | 70.74 / 1.262 / 1.059  | 0.3 hrs. (1 RTX3090) |
| R56-BN-ReLU      | 99.72 / 0.022 / 0.116  | 71.34 / 1.302 / 1.055  | 71.80 / 1.278 / 1.043  | 0.4 hrs. (1 RTX3090) |
| WRN28x1-BN-ReLU  | 96.43 / 0.149 / 0.269  | 69.42 / 1.232 / 1.106  | 68.86 / 1.262 / 1.131  | 0.2 hrs. (1 RTX3090) |
| WRN28x5-BN-ReLU  | 99.99 / 0.001 / 0.003  | 78.62 / 0.872 / 0.860  | 78.91 / 0.869 / 0.858  | 0.7 hrs. (2 RTX3090) |
| WRN28x10-BN-ReLU | 99.99 / 0.001 / 0.002  | 80.38 / 0.813 / 0.809  | 79.72 / 0.826 / 0.822  | 2.1 hrs. (2 RTX3090) |
|                  | 99.98 / 0.001 / 0.003  | 80.28 / 0.843 / 0.826  | 80.78 / 0.805 / 0.795  | 1.7 hrs. (4 RTX3090) |
