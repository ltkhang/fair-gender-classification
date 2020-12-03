from src.evals import cal_acc
import sys
import os

test_name = {
    'fairface': "test,test_Black,test_East Asian,test_Indian,test_Latino_Hispanic,test_Middle Eastern,test_Southeast Asian,test_White",
    'utkface': "test,test_Asian,test_Black,test_Indian,test_Others,test_White",
    'adience': "fold_0,fold_1,fold_2,fold_3,fold_4"
}
ds_path = {
    'fairface': './rec/fairface',
    'utkface': './rec/utkface',
    'adience': './rec/adience'
}

if __name__ == '__main__':
    batch_size = 64
    print(sys.argv)
    checkpoint_path = sys.argv[1]
    epoch = int(sys.argv[2])
    path = ds_path[sys.argv[3]]
    list_test_name = [n for n in test_name[sys.argv[3]].split(',')]
    log_file = sys.argv[4]
    log_file = os.path.join('./log', log_file + '.txt')
    res = cal_acc(checkpoint_path, epoch, batch_size, path, list_test_name)
    f = open(log_file, 'w')
    for i in range(len(res)):
        print(list_test_name[i], ':', res[i])
        f.write('{} : {}\n'.format(list_test_name[i], res[i]))
    f.close()