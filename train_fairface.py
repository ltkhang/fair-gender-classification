from src.train import train
from src.data import get_train_val_test, get_data
import sys


if __name__ == '__main__':
    batch_size = 64
    print(sys.argv)
    model_name = sys.argv[1]
    dataset = sys.argv[2]
    epoch = int(sys.argv[3])
    if dataset == 'fairface' or dataset == 'utkface':
        train_set, val_set, test_set = get_train_val_test(batch_size, 'rec/{}'.format(dataset))
    else:
        train_path = sys.argv[4]
        val_path = sys.argv[5]
        train_set = get_data(batch_size, train_path, True)
        val_set = test_set = get_data(batch_size, val_path)
    train(model_name, dataset, train_set, val_set, test_set, batch_size, epoch=epoch)