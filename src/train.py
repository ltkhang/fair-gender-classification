import mxnet as mx
from .symbol import get_resnet34, get_balance, get_ms1m
from .data import get_data, get_train_val_test, get_data_list
import os

import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

NUM_CLASSES = 2


def fit(epoch, symbol, arg_params, aux_params, train, val, test, batch_size, checkpoint, steps=None,
        fixed_param_names=None):
    # devs = [mx.gpu(i) for i in range(num_gpus)] # uncomment for training on multiple gpu
    devs = [mx.gpu(0)]
    mod = mx.mod.Module(symbol=symbol, context=devs, fixed_param_names=fixed_param_names)
    if steps is not None:
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(steps, 0.1)
        optimizer_params = {'learning_rate': 0.01, 'lr_scheduler': lr_scheduler}
    else:
        optimizer_params = {'learning_rate': 0.01}
    mod.fit(train, val,
            num_epoch=epoch,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer='sgd',
            optimizer_params=optimizer_params,
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc',
            epoch_end_callback=mx.callback.do_checkpoint(checkpoint))
    metric = mx.metric.Accuracy()
    score = mod.score(test, metric)
    devs[0].empty_cache()
    return score


def train(backbone, dataset_name, train_set, val_set, test_set, batch_size, epoch=20):
    tmp_checkpoint = os.path.join('tmp_checkpoints', backbone)
    if not os.path.exists(tmp_checkpoint):
        os.mkdir(tmp_checkpoint)
    tmp_checkpoint = os.path.join(tmp_checkpoint, dataset_name)
    if not os.path.exists(tmp_checkpoint):
        os.mkdir(tmp_checkpoint)
    print('Train', backbone, 'on dataset', dataset_name)
    # if backbone == 'resnet34':
    #     sym, arg_params, aux_params = mx.model.load_checkpoint('../models/modelresnet34/model', 0)
    #     new_sym, new_args = get_resnet34(sym, arg_params, NUM_CLASSES)
    # elif backbone == 'ms1m':
    #     sym, arg_params, aux_params = mx.model.load_checkpoint('../models/modelms1m/model', 0)
    #     new_sym, new_args = get_ms1m(sym, arg_params, NUM_CLASSES)
    # else:  # balance
    #     sym, arg_params, aux_params = mx.model.load_checkpoint('../models/modelbalance/model', 0)
    #     new_sym, new_args = get_balance(sym, arg_params, NUM_CLASSES)

    sym, arg_params, aux_params = mx.model.load_checkpoint('models/model{}/model'.format(backbone), 0)
    new_sym, new_args = eval('get_{}'.format(backbone))(sym, arg_params, NUM_CLASSES)

    mod_score = fit(epoch, new_sym, new_args, aux_params, train_set, val_set, test_set, batch_size,
                    os.path.join(tmp_checkpoint, 'model'),
                    fixed_param_names=None)
    print('Test accuracy', mod_score)
