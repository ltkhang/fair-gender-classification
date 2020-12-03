from src.evals import find_best, cal_acc
import sys
import os


if __name__ == '__main__':
    batch_size = 64
    print(sys.argv)
    checkpoint_path = sys.argv[1]
    epoch = int(sys.argv[2])
    path = sys.argv[3]
    list_test_name = [n for n in sys.argv[4].split(',')]
    print(list_test_name)
    log_file = sys.argv[5]
    log_file = os.path.join('./log', log_file)
    res_epoch, res = find_best(checkpoint_path, epoch, batch_size, path, list_test_name)
    print(cal_acc(checkpoint_path, res_epoch[0], batch_size, path, list_test_name))
    f = open(log_file, 'w')
    for i in range(len(res)):
        print(list_test_name[i], '(', res_epoch[i], ')', ':', res[i])
        f.write('{} ({}) : {}\n'.format(list_test_name[i], res_epoch[i], res[i]))
    f.close()
