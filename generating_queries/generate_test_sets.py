# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

from time import sleep
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse

# For training and test data splits
X_WIDTH = 150
Y_WIDTH = 150

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

P_DICT = {"oxford": [P1, P2, P3, P4], "university": [P5, P6, P7], "residential": [P8, P9, P10], "business": []}


def check_in_test_set(northing, easting, points):
    """
    # 检查northing和easting是否在points的附近[150m, 150m]的范围内
    """
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set


def output_to_file(output, base_path, filename):
    # file_path = os.path.join(base_path, filename)
    # with open(file_path, 'wb') as handle:
    #     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("Done ", filename)
    print("Do not write!")

def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    """
    # 做一个database的测试集database和query的pickle文件
    ## input params:\n
    * basepath: 数据集根目录 /nas/slam/.../DataMinkLoc3D
    * runs_folders: [oxford, inhouse_datasets]
    * baseline的traning只包括oxford
    * folders: 是一个list 例如 [residential_run1, residential_run2]   # baseline的traning只包括oxford
    * pointcloud_fols: 点云类型
    * filename: csv文件名
    * p: 
    """
    database_trees = []
    test_trees = []
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            """
            row = {
                timestamp    1.415985e+15
                northing     5.735636e+06
                easting      6.200410e+05
                Name: 6, dtype: float64
            }
            """
            # 符合条件的子图才能被加入df_test
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
                # df_test = pd.concat([df_test, row], ignore_index=True)
            # 除了business，其余子图要在p的附近才能加入df_test
            elif check_in_test_set(row['northing'], row['easting'], p):
                # df_test = pd.concat([df_test, row], ignore_index=True)
                df_test = df_test.append(row, ignore_index=True)

            # 报warn
            # FutureWarning: The frame.append method is deprecated and will 
            # be removed from pandas in a future version. Use pandas.concat instead.

            # 所有子图都能加入df_database
            df_database = df_database.append(row, ignore_index=True)
            # df_database = pd.concat([df_database, row], ignore_index=True)

        
        # 一个序列的database_tree和test_tree
        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        # 一个database的所有kdtree
        database_trees.append(database_tree)
        test_trees.append(test_tree)
    # query
    test_sets = []
    # database
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        # 读csv文件
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # 获取时间戳
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + \
                                    df_locations['timestamp'].astype(str) + '.bin'
        # 把timestamp改名为file, 就是子图.bin的文件名
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            # 整个business都在test set范围内
            # test.keys(): [0, 1, 2, ..., n]
            # len(test.keys()) 是test当前元素个数
            """
            row:
            file        oxford/2014-11-14-16-34-33/pointcloud_20m/bin文件名
            northing                                       5735709.234639
            easting                                         620007.466377
            Name: 2, dtype: object
            """
            # 同样符合条件的才能加入test
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            # 所有都加入database
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                                'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            # 排除自己
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                # 找到25m以内的其它索引
                index = tree.query_radius(coor, r=25)
                # print(index[0].tolist())
                # [392, 14, 15, 391, 390]
                # indices of the positive matches in database i of each query (key) in test set j
                # 找到database[i]中与test_sets[j][key]的为正匹配的索引
                test_sets[j][key][i] = index[0].tolist()


    output_to_file(database_sets, base_path, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path, output_name + '_evaluation_query.pickle')


if __name__ == '__main__':
    """
    self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                'residential_evaluation_database.pickle', 'university_evaluation_database.pickle']

    self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']
    """
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    # For Oxford
    folders = []
    runs_folder = "oxford/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # 对于Oxford 抽一些序列的索引来测试
    index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/",
                                      "pointcloud_locations_20m.csv", P_DICT["oxford"], "oxford")

    # For University Sector
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    uni_index = range(10, 15)
    for index in uni_index:
        folders.append(all_folders[index])

    print(folders)
    # pointcloud_25m_25代表了basline
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["university"], "university")

    # For Residential Area
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    res_index = range(5, 10)
    for index in res_index:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["residential"], "residential")

    # For Business District
    folders = []
    runs_folder = "inhouse_datasets/"
    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    bus_index = range(5)
    for index in bus_index:
        folders.append(all_folders[index])

    print(folders)
    # folders 是business database的所有文件夹名
    # folders = businiess_run 1~5
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
                                      "pointcloud_centroids_25.csv", P_DICT["business"], "business")
