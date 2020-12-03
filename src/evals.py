import mxnet as mx
from .data import get_data, get_train_val_test, get_data_list
import os
import statistics


def evaluate(symbol, arg_params, aux_params, test):
    # devs = [mx.gpu(i) for i in range(num_gpus)]
    devs = [mx.gpu(0)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.bind(data_shapes=test.provide_data, label_shapes=test.provide_label)
    mod.set_params(arg_params, aux_params)
    metric = mx.metric.Accuracy()
    return mod.score(test, metric)[0][1]


def find_best(checkpoint_path, epoch, batch_size, path, list_test_name):
    test_list = get_data_list(batch_size, path, list_test_name)
    res = []
    res_epoch = []
    for test in test_list:
        best = 0.0
        best_epoch = 0
        for e in range(epoch):
            idx = e + 1
            sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_path, idx)
            score = evaluate(sym, arg_params, aux_params, test)
            if score > best:
                best = score
                best_epoch = idx
        res.append(best)
        res_epoch.append(best_epoch)
    return res_epoch, res


def cal_acc(checkpoint_path, epoch, batch_size, path, list_test_name):
    test_list = get_data_list(batch_size, path, list_test_name)
    res = []
    for test in test_list:
        sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_path, epoch)
        score = evaluate(sym, arg_params, aux_params, test)
        res.append(score)
    return res
