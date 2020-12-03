import mxnet as mx


def get_data(batch_size, path, aug=False, data_shape=(3, 112, 112)):
    if aug:
        data = mx.io.ImageRecordIter(
            path_imglist='{}.lst'.format(path),
            path_imgidx='{}.idx'.format(path),
            path_imgrec='{}.rec'.format(path),
            data_name='data',
            label_name='softmax_label',
            batch_size=batch_size,
            data_shape=data_shape,
            shuffle=aug,
            rand_mirror=aug,
            brightness=0.6,
            contrast=0.6)
    else:
        data = mx.io.ImageRecordIter(
            path_imglist='{}.lst'.format(path),
            path_imgidx='{}.idx'.format(path),
            path_imgrec='{}.rec'.format(path),
            data_name='data',
            label_name='softmax_label',
            batch_size=batch_size,
            data_shape=data_shape)
    return data


def get_train_val_test(batch_size, path):
    train = get_data(batch_size, path + '/train', True)
    val = get_data(batch_size, path + '/val')
    test = get_data(batch_size, path + '/test')
    return train, val, test


def get_data_list(batch_size, path, list_name):
    data = []
    for name in list_name:
        data.append(get_data(batch_size, path + '/' + name))
    return data
