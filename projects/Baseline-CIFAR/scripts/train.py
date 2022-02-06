import os
import sys
import shutil
import logging
import datetime
import argparse
import functools
from tabulate import tabulate
from typing import Any
sys.path.append('./')

import jax
import jaxlib
import flax
import optax
from jax import numpy as jnp
from flax import jax_utils, serialization
from flax.training import train_state, checkpoints
from flax.training.common_utils import get_metrics, onehot
from tensorflow.io import gfile

from giung2.config import get_cfg
from giung2.modeling.build import build_model
from giung2.data.build import build_dataloaders


class TrainState(train_state.TrainState):
    image_stats: Any = None
    batch_stats: Any = None


def step_trn(state, batch, num_classes, scheduler):

    # define loss function
    def loss_fn(params):
        output, new_model_state = state.apply_fn(
            {
                'params': params,
                'batch_stats': state.batch_stats,
                'image_stats': state.image_stats,
            }, batch['images'], mutable=['batch_stats',],
            use_running_average=False,
        )

        # loss_ce
        target = onehot(batch['labels'], num_classes=num_classes)
        loss_ce = jnp.mean(-jnp.sum(output * target, axis=-1))

        # loss_wd
        wd_params = jax.tree_leaves(params)
        loss_wd = 0.5 * sum([
            jnp.sum(e**2) for e in wd_params # if e.ndim > 1
        ])

        loss = loss_ce + args.weight_decay * loss_wd
        metrics = {
            'acc1': jnp.mean(jnp.argmax(output, -1) == batch['labels']),
            'loss': loss,
            'loss_ce': loss_ce,
            'loss_wd': loss_wd,
        }
        return loss, (metrics, new_model_state)

    # define grad function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, aux), grad = grad_fn(state.params)
    metrics, new_model_state = aux

    # update model
    grad = jax.lax.pmean(grad, axis_name='batch')
    new_state = state.apply_gradients(
        grads       = grad,
        batch_stats = new_model_state['batch_stats'],
    )

    metrics = jax.lax.pmean(metrics, axis_name='batch')
    metrics['lr'] = scheduler(state.step)

    return new_state, metrics


def step_val(state, batch, num_classes):
    output = state.apply_fn(
        {
            'params': state.params,
            'batch_stats': state.batch_stats,
            'image_stats': state.image_stats,
        }, batch['images'], mutable=False,
        use_running_average=True,
    )
    target = onehot(batch['labels'], num_classes=num_classes)
    metrics = {
        'acc1': jnp.mean(jnp.argmax(output, -1) == batch['labels']),
        'loss_ce': jnp.mean(-jnp.sum(output * target, axis=-1))
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, required=True, metavar='FILE',
                        help='path to config file')
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='number of epochs for training')
    parser.add_argument('--num_warmup_epochs', default=5, type=int,
                        help='number of initial epochs for learning rate warm-up')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='number of examples per one mini-batch')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='base learning rate for training')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay coefficient')
    parser.add_argument('--output_dir', default=None, required=True,
                        help='output directory')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed for training')
    parser.add_argument('--keep_every_n_checkpoints', default=None, type=int,
                        help='if specified, keep every checkpoints every n epochs')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config options at the end of the command')
    args = parser.parse_args()

    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # define logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    logger_s_handler = logging.StreamHandler(stream=sys.stdout)
    logger_s_handler.setLevel(logging.INFO)
    logger_s_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_s_handler)

    log = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logger_f_handler = logging.FileHandler(os.path.join(args.output_dir, f'{log}.log'))
    logger_f_handler.setLevel(logging.INFO)
    logger_f_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_f_handler)
    logger_f_handler = logging.FileHandler(os.path.join(args.output_dir, f'{log}.debug.log'))
    logger_f_handler.setLevel(logging.DEBUG)
    logger_f_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_f_handler)

    # log environments
    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__ + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__ + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    logger.info(f'Environments:\n{log_str}')

    # log command line arguments
    log_str = str(args) + '\n'
    logger.info(f'Command line arguments:\n{log_str}')

    # log the random seed
    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )
    logger.info(f'Using a random seed {args.seed}\n')

    # initialize configuration
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, allow_unsafe=True)
    cfg.merge_from_list(args.opts)

    # build model
    model = build_model(cfg)
    rng = jax.random.PRNGKey(args.seed)

    # initialize model    
    def initialize_model(key, model, im_shape, im_dtype):
        @jax.jit
        def init(*args):
            return model.init(*args)
        var_dict = init({'params': key}, jnp.ones(im_shape, im_dtype))
        return var_dict

    if cfg.DATASETS.NAME in ['CIFAR10',]:
        image_shape = (1, 32, 32, 3,)
        num_classes = 10
    elif cfg.DATASETS.NAME in ['CIFAR100',]:
        image_shape = (1, 32, 32, 3,)
        num_classes = 100
    else:
        raise NotImplementedError

    im_dtype = jnp.float32
    var_dict = initialize_model(rng, model, image_shape, im_dtype)

    # build dataset
    dataloaders = build_dataloaders(cfg, batch_size=[args.batch_size, 200, 200])
    trn_steps_per_epoch = dataloaders['trn_steps_per_epoch']

    # build optimizer
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value       = 0.0,
                end_value        = args.learning_rate,
                transition_steps = args.num_warmup_epochs * trn_steps_per_epoch,
            ),
            optax.cosine_decay_schedule(
                init_value       = args.learning_rate,
                decay_steps      = (args.num_epochs - args.num_warmup_epochs) * trn_steps_per_epoch,
            )
        ], boundaries=[args.num_warmup_epochs * trn_steps_per_epoch,]
    )
    optimizer = optax.sgd(learning_rate=scheduler, momentum=0.9, nesterov=True)

    # build trainer
    state = TrainState.create(
        apply_fn    = model.apply,
        params      = var_dict['params'],
        tx          = optimizer,
        image_stats = var_dict['image_stats'],
        batch_stats = var_dict['batch_stats'] if 'batch_stats' in var_dict else {},
    )
    state = checkpoints.restore_checkpoint(
        ckpt_dir    = args.output_dir,
        target      = state,
        step        = None,
        prefix      = 'ckpt_e',
        parallel    = True,
    )
    epoch_offset = int(state.step) // trn_steps_per_epoch

    # train model
    step_trn = jax.pmap(functools.partial(step_trn, num_classes=num_classes, scheduler=scheduler), axis_name='batch')
    step_val = jax.pmap(functools.partial(step_val, num_classes=num_classes                     ), axis_name='batch')
    sync_batch_stats = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')

    best_acc1 = 0.0
    best_loss = float('inf')
    best_acc1_path = os.path.join(args.output_dir, 'best_acc1')
    best_loss_path = os.path.join(args.output_dir, 'best_loss')

    state = jax_utils.replicate(state)
    for epoch_idx in range(epoch_offset + 1, args.num_epochs + 1):

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        rng, data_rng = jax.random.split(rng)

        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=data_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader, start=1):
            state, metrics = step_trn(state, batch)
            trn_metric.append(metrics)

        trn_metric = get_metrics(trn_metric)
        summarized = {f'trn/{k}': v for k, v in jax.tree_map(lambda e: e.mean(), trn_metric).items()}
        logger.info(
            f'[Epoch {epoch_idx:3d}/{args.num_epochs:3d}] '
            + ', '.join(f'{k} {v:.2e}' for k, v in summarized.items())
        )

        # synchronize batch statitics across replicas
        if state.batch_stats:
            state = state.replace(batch_stats=sync_batch_stats(state.batch_stats))

        # save checkpoint from the first replica
        if args.keep_every_n_checkpoints and jax.process_index() == 0:
            _state = jax.device_get(jax.tree_map(lambda x: x[0], state))
            checkpoints.save_checkpoint(
                ckpt_dir           = args.output_dir,
                target             = _state,
                step               = epoch_idx, # _state.step,
                prefix             = 'ckpt_e',
                keep               = 3,
                overwrite          = False,
                keep_every_n_steps = args.keep_every_n_checkpoints,
            )

        # ---------------------------------------------------------------------- #
        # Valid
        # ---------------------------------------------------------------------- #
        if (epoch_idx == 1
            or epoch_idx in list(range(int(args.num_epochs * 0.1), args.num_epochs, int(args.num_epochs * 0.1)))
            or epoch_idx >= int(args.num_epochs * 0.9)):

            val_metric = []
            val_loader = dataloaders['val_loader'](rng=None)
            val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
            for batch_idx, batch in enumerate(val_loader, start=1):
                metrics = step_val(state, batch)
                val_metric.append(metrics)

            val_metric = get_metrics(val_metric)
            summarized = {f'val/{k}': v for k, v in jax.tree_map(lambda e: e.mean(), val_metric).items()}
            logger.info(
                f'[Epoch {epoch_idx:3d}/{args.num_epochs:3d}] Validation: '
                + ', '.join(f'{k} {v:.4f}' for k, v in summarized.items())
            )

            # save checkpoint from the first replica
            if jax.process_index() == 0:
                ckptpath = os.path.join(args.output_dir, f'ckpt_e{epoch_idx}')

                if not os.path.exists(ckptpath):
                    _state = jax.device_get(jax.tree_map(lambda x: x[0], state))
                    with gfile.GFile(ckptpath, 'wb') as fp:
                        fp.write(serialization.to_bytes(_state))

                if summarized['val/acc1'] > best_acc1:
                    best_acc1 = summarized['val/acc1']
                    shutil.copyfile(ckptpath, best_acc1_path)
                
                if summarized['val/loss_ce'] < best_loss:
                    best_loss = summarized['val/loss_ce']
                    shutil.copyfile(ckptpath, best_loss_path)
