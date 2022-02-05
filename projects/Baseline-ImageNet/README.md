# ImageNet Baselines

## TinyImageNet

In summary,
* Use 90,000 train examples, 10,000 valid examples, and 10,000 test examples from 200 classes.
* Use train data augmentation consisting of random cropping with padding and random horizontal flipping.
* Use SGD optimizer with Nesterov momentum 0.9, batch size 128, and base learning rate 0.1.
* Use single-cycle cosine annealed learning rate schedule with a linear warm-up.

Run the following command lines:
```
python scripts/train.py \
    --config_file ./configs/TIN200_R18-BN-ReLU_SGD.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/TIN200_R18-BN-ReLU/SGD/s42_e100_wd5e-4/
```
```
python scripts/train.py \
    --config_file ./configs/TIN200_R50-BN-ReLU_SGD.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/TIN200_R50-BN-ReLU/SGD/s42_e100_wd5e-4/
```

As a result, we get the following:
| Network      | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        |
| :-           | :-:                    | :-:                    | :-:                    | :-:                  |
| R18-BN-ReLU  | 99.98 / 0.007 / 0.016  | 65.71 / 1.477 / 1.458  | 65.47 / 1.516 / 1.491  | 1.0 hrs. (2 RTX3090) |
| R50-BN-ReLU  | 99.98 / 0.004 / 0.008  | 69.64 / 1.314 / 1.300  | 68.99 / 1.348 / 1.331  | 3.5 hrs. (2 RTX3090) |
| R18-FRN-SiLU | 99.95 / 0.032 / 0.057  | 63.62 / 1.575 / 1.560  | 63.48 / 1.589 / 1.574  | 1.0 hrs. (2 RTX3090) |

## ImageNet

(TBD)
