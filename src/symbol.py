import mxnet as mx


def get_resnet34(symbol, arg_params, num_classes):
    layer_name = 'flatten0'
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args


def get_balance(symbol, arg_params, num_classes):
    layer_name = 'fc1'
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    all_layers = symbol.get_internals()
    embedding = all_layers[layer_name + '_output']
    net = mx.symbol.Flatten(data=embedding, name='flatten_1')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc7')
    net = mx.symbol.SoftmaxOutput(data=net, label=gt_label, name='softmax', normalization='valid')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc7' not in k})
    return net, new_args


def get_ms1m(symbol, arg_params, num_classes):
    layer_name = 'fc1'
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    all_layers = symbol.get_internals()
    embedding = all_layers[layer_name + '_output']
    net = mx.symbol.Flatten(data=embedding, name='flatten_1')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc7')
    net = mx.symbol.SoftmaxOutput(data=net, label=gt_label, name='softmax', normalization='valid')
    return net, arg_params
