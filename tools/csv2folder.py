import pandas as pd
import os
from shutil import copyfile


def csv2folder(csv_files_list, data_dir, dst_dir, name=None):
    df_list = []
    for csv_file in csv_files_list:
        df_list.append(pd.read_csv(csv_file))
    df = pd.concat(df_list, ignore_index=True)
    if name is not None:
        dst_dir = os.path.join(dst_dir, name)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    gender_list = ['Male', 'Female']
    for gender in gender_list:
        gender_path = os.path.join(dst_dir, gender)
        if not os.path.exists(gender_path):
            os.mkdir(gender_path)

    for i in range(len(df)):
        file = df.loc[i, 'file']
        gender = df.loc[i, 'gender']
        src = os.path.join(data_dir, file)
        gender_path = os.path.join(dst_dir, gender)
        new_file = file.replace('/', '_')
        dst = os.path.join(gender_path, new_file)
        print(file, gender, '-->', dst)
        copyfile(src, dst)


if __name__ == '__main__':
    # adience_base = '../../GenderDataset/Adience'
    # adience_data_dir = os.path.join(adience_base, 'adience')
    # adience_csv_dir = os.path.join(adience_base, 'csv')
    # adience_dst_dir = '../raw_data/adience'
    # for i in range(5):
    #     fold_name = 'fold_{}'.format(i)
    #     csv2folder([os.path.join(adience_csv_dir, fold_name + '.csv')], adience_data_dir, adience_dst_dir, fold_name)
    # for i in range(5):
    #     fold_name = ''
    #     csv_list = []
    #     for j in range(5):
    #         if j != i:
    #             fold_name += 'fold_{}'.format(j)
    #             csv_list.append(os.path.join(adience_csv_dir, 'fold_{}.csv'.format(j)))
    #     csv2folder(csv_list, adience_data_dir, adience_dst_dir, fold_name)
    # fairface_base = '../../GenderDataset/FairFace'
    # fairface_data_dir = os.path.join(fairface_base, 'fairface')
    # fairface_csv_dir = os.path.join(fairface_base, 'annotation')
    # fairface_dst_dir = '../raw_data/fairface'
    # csv_list = ['test', 'train', 'val', 'test_Black', 'test_East Asian', 'test_Indian', 'test_Latino_Hispanic',
    #             'test_Middle Eastern', 'test_Southeast Asian', 'test_White']
    # for csv_name in csv_list:
    #     csv_path = os.path.join(fairface_csv_dir, '{}.csv'.format(csv_name))
    #     csv2folder([csv_path], fairface_data_dir, fairface_dst_dir, csv_name)
    # utkface_base = '../../GenderDataset/UTKFace'
    # utkface_data_dir = os.path.join(utkface_base, 'utkface')
    # utkface_csv_dir = os.path.join(utkface_base, 'csv')
    # utkface_dst_dir = '../raw_data/utkface'
    # csv_list = ['test', 'train', 'val', 'test_Asian', 'test_Black', 'test_Indian', 'test_Others', 'test_White']
    # for csv_name in csv_list:
    #     csv_path = os.path.join(utkface_csv_dir, '{}.csv'.format(csv_name))
    #     csv2folder([csv_path], utkface_data_dir, utkface_dst_dir, csv_name)
    print('Done!')


