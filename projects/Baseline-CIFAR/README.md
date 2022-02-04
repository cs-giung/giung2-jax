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
    --config_file ./configs/C10_WRN28x10-BN-ReLU_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C10_WRN28x5-BN-ReLU/SGD/s42_e200_wd5e-4/ \
    MODEL.BACKBONE.RESNET.WIDEN_FACTOR 5
```
```
python scripts/train.py \
    --config_file ./configs/C10_WRN28x10-BN-ReLU_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C10_WRN28x10-BN-ReLU/SGD/s42_e200_wd5e-4/
```

As a result, we get the following:
| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  |
| WRN28x5-BN-ReLU  |
| WRN28x10-BN-ReLU |

## CIFAR-100

Run the following command lines:
```
python scripts/train.py \
    --config_file ./configs/C100_WRN28x10-BN-ReLU_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C100_WRN28x5-BN-ReLU/SGD/s42_e200_wd5e-4/ \
    MODEL.BACKBONE.RESNET.WIDEN_FACTOR 5
```
```
python scripts/train.py \
    --config_file ./configs/C100_WRN28x10-BN-ReLU_SGD.yaml \
    --num_epochs 200 --num_warmup_epochs 5 \
    --batch_size 128 --learning_rate 0.1 --weight_decay 5e-4 \
    --seed 42 --output_dir ./outputs/C100_WRN28x10-BN-ReLU/SGD/s42_e200_wd5e-4/
```

As a result, we get the following:
| Network          | Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  | Train Runtime        |
| :-               | :-:                    | :-:                    | :-:                    | :-:                  |
| WRN28x5-BN-ReLU  |
| WRN28x10-BN-ReLU |
