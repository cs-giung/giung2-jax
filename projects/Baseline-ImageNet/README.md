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
    --config_file ./configs/TIN200_{NETWORK_NAME}_SGD.yaml \
    --num_epochs 100 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/TIN200_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/
```
```
python scripts/eval.py \
    --config_file ./configs/TIN200_{NETWORK_NAME}_SGD.yaml \
    --weight_file ./outputs/TIN200_{NETWORK_NAME}/SGD/s42_e100_wd5e-4/best_acc1
```

As a result, we get the following:
| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        | Misc. |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  | :-:   |
| R18-BN-ReLU      | 99.98 / 0.009 / 0.017  | 64.98 / 1.497 / 1.485  | 64.98 / 1.533 / 1.519  | 1.6 hrs. (1 RTX3090) | [log](./scripts/logs/TIN200/20220206151326.log) |
|                  | 99.98 / 0.007 / 0.018  | 65.84 / 1.488 / 1.464  | 65.44 / 1.529 / 1.499  | 1.1 hrs. (2 RTX3090) | [log](./scripts/logs/TIN200/20220205132411.log) |
|                  | 99.98 / 0.006 / 0.018  | 66.48 / 1.480 / 1.439  | 65.51 / 1.547 / 1.495  | 0.8 hrs. (4 RTX3090) | [log](./scripts/logs/TIN200/20220205121009.log) |
|                  | 99.98 / 0.005 / 0.022  | 66.07 / 1.544 / 1.459  | 65.52 / 1.581 / 1.486  | 0.6 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220206063249.log) |
| R50-BN-ReLU      | 99.98 / 0.005 / 0.008  | 69.73 / 1.305 / 1.300  | 69.20 / 1.326 / 1.321  | 5.7 hrs. (1 RTX3090) | [log](./scripts/logs/TIN200/20220206151219.log) |
|                  | 99.98 / 0.004 / 0.009  | 69.53 / 1.299 / 1.288  | 69.16 / 1.342 / 1.327  | 3.5 hrs. (2 RTX3090) | [log](./scripts/logs/TIN200/20220205132437.log) |
|                  | 99.98 / 0.003 / 0.010  | 70.10 / 1.309 / 1.276  | 69.22 / 1.353 / 1.315  | 2.2 hrs. (4 RTX3090) | [log](./scripts/logs/TIN200/20220205120813.log) |
|                  | 99.97 / 0.004 / 0.016  | 69.80 / 1.343 / 1.265  | 69.36 / 1.378 / 1.296  | 1.1 hrs. (8 TPUv3)   | [log](./scripts/logs/TIN200/20220206071326.log) |

## ImageNet

(TBD)
