import os
import sys
import argparse
import functools
from tqdm import tqdm
sys.path.append('./')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF logs

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from flax import jax_utils
from flax.training import checkpoints
from flax.training.common_utils import onehot

from giung2.config import get_cfg
from giung2.data.build import build_dataloaders
from giung2.modeling.build import build_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, required=True, metavar='FILE',
                        help='path to config file')
    parser.add_argument('--weight_file', default=None, required=True, metavar='FILE',
                        help='path to weight file')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='number of examples per one mini-batch')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config options at the end of the command')
    args = parser.parse_args()

    # initialize configuration
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, allow_unsafe=True)
    cfg.merge_from_list(args.opts)

    # build model
    model = build_model(cfg)
    rng = jax.random.PRNGKey(0)

    # initialize model    
    def initialize_model(key, model, im_shape, im_dtype):
        @jax.jit
        def init(*args):
            return model.init(*args)
        var_dict = init({'params': key}, jnp.ones(im_shape, im_dtype))
        return var_dict

    USEIMAGENET = False
    if cfg.DATASETS.NAME in ['TinyImageNet200',]:
        image_shape = (1, 64, 64, 3,)
        num_classes = 200
    elif cfg.DATASETS.NAME in ['ImageNet1k_x32',]:
        image_shape = (1, 32, 32, 3,)
        num_classes = 1000
    elif cfg.DATASETS.NAME in ['ImageNet1k_x64',]:
        image_shape = (1, 64, 64, 3,)
        num_classes = 1000
    elif cfg.DATASETS.NAME in ['ImageNet1k',]:
        image_shape = (1, 224, 224, 3,)
        num_classes = 1000
        USEIMAGENET = True
    else:
        raise NotImplementedError

    im_dtype = jnp.float32
    var_dict = initialize_model(rng, model, image_shape, im_dtype)

    # load pre-trained weights
    ckpt = checkpoints.restore_checkpoint(args.weight_file, target=None)
    var_dict = {
        'params': ckpt['params'],
        'image_stats': ckpt['image_stats'],
        'batch_stats': ckpt['batch_stats'],
    }

    # define functions
    CPU = jax.devices("cpu")[0]

    def predict(var_dict, images):
        _, outputs = model.apply(var_dict, images, mutable='intermediates')
        return outputs['intermediates']['classifier']['log_confidences'][0]

    predict = jax.pmap(functools.partial(predict, var_dict), axis_name='batch')

    def make_predictions(dataloader, desc, num_steps_per_epoch=None):
        true_labels, pred_lconfs = [], []
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=desc, leave=False):
            labels = jax.device_put(jnp.concatenate(        batch['labels'] ), CPU)
            lconfs = jax.device_put(jnp.concatenate(predict(batch['images'])), CPU)
            true_labels.append(labels)
            pred_lconfs.append(lconfs)
            if (num_steps_per_epoch is not None
                and batch_idx == num_steps_per_epoch):
                break
        return jnp.concatenate(true_labels), jnp.concatenate(pred_lconfs)

    @jax.jit
    def evaluate_acc(log_confidences, true_labels):
        return jnp.mean(jnp.argmax(log_confidences, axis=1) == true_labels)

    @jax.jit
    def evaluate_nll(log_confidences, true_labels):
        return jnp.mean(-jnp.sum(log_confidences * onehot(true_labels, num_classes=log_confidences.shape[1]), axis=-1))

    @jax.jit
    def temperature_scaling(log_confidences, temperature):
        return jax.nn.log_softmax(log_confidences / temperature, axis=-1)

    @jax.jit
    def get_optimal_temperature(log_confidences, true_labels):
        def obj(t):
            return evaluate_nll(temperature_scaling(log_confidences, t), true_labels)
        optimal_temperature = minimize(obj, jnp.asarray([1.0,]), method='BFGS', tol=1e-3).x[0]
        return optimal_temperature

    if USEIMAGENET:

        # build dataset
        from scripts.input_pipeline import build_imagenet_dataloader
        dataloaders = build_imagenet_dataloader(batch_size=args.batch_size)
        val_steps_per_epoch = dataloaders['val_steps_per_epoch']

        # make predictions
        val_true_labels, val_pred_lconfs = make_predictions(
            jax_utils.prefetch_to_device(dataloaders['val_loader'](rng=None), size=2),
            'Make predictions on valid examples',
            val_steps_per_epoch)

        # evaluate predictions
        t_opt = get_optimal_temperature(val_pred_lconfs, val_true_labels)
        print('| Valid ACC / NLL / cNLL |')
        print(
            '| ' +
            ('%.2f' % (100 * evaluate_acc(                    val_pred_lconfs,         val_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(                    val_pred_lconfs,         val_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(temperature_scaling(val_pred_lconfs, t_opt), val_true_labels)))[:5] + '  | ' +
            '\n'
        )

    else:

        # build dataset
        dataloaders = build_dataloaders(cfg, batch_size=args.batch_size)

        # make predictions
        trn_true_labels, trn_pred_lconfs = make_predictions(
            jax_utils.prefetch_to_device(dataloaders['trn_loader'](rng=None), size=2),
            'Make predictions on train examples')
        val_true_labels, val_pred_lconfs = make_predictions(
            jax_utils.prefetch_to_device(dataloaders['val_loader'](rng=None), size=2),
            'Make predictions on valid examples')
        tst_true_labels, tst_pred_lconfs = make_predictions(
            jax_utils.prefetch_to_device(dataloaders['tst_loader'](rng=None), size=2),
            'Make predictions on test examples')

        # evaluate predictions
        t_opt = get_optimal_temperature(val_pred_lconfs, val_true_labels)
        print('| Train ACC / NLL / cNLL | Valid ACC / NLL / cNLL | Test ACC / NLL / cNLL  |')
        print(
            '| ' +
            ('%.2f' % (100 * evaluate_acc(                    trn_pred_lconfs,         trn_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(                    trn_pred_lconfs,         trn_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(temperature_scaling(trn_pred_lconfs, t_opt), trn_true_labels)))[:5] + '  | ' +
            ('%.2f' % (100 * evaluate_acc(                    val_pred_lconfs,         val_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(                    val_pred_lconfs,         val_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(temperature_scaling(val_pred_lconfs, t_opt), val_true_labels)))[:5] + '  | ' +
            ('%.2f' % (100 * evaluate_acc(                    tst_pred_lconfs,         tst_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(                    tst_pred_lconfs,         tst_true_labels)))[:5] + ' / ' +
            ('%.3f' % (      evaluate_nll(temperature_scaling(tst_pred_lconfs, t_opt), tst_true_labels)))[:5] + '  | ' +
            '\n'
        )
