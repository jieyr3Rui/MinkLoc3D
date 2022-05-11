# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from time import sleep
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import tqdm
import MinkowskiEngine as ME
import random

from misc.utils import MinkLocParams
from models.model_factory import model_factory

# 导入ValTransform 
from datasets.oxford import ValTransform

def evaluate(model, device, params, silent=True):
    """
    # 测试所有数据集

    self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                'residential_evaluation_database.pickle', 'university_evaluation_database.pickle']

    self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']

    """
    # Run evaluation on all eval datasets
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets, silent=silent)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    print("computing database_embeddings ...")
    for set in tqdm.tqdm(database_sets, disable=silent):
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    print("computing query_embeddings ...")
    for set in tqdm.tqdm(query_sets, disable=silent):
        query_embeddings.append(get_latent_vectors(model, set, device, params))


    # 只迭代query_sets 自身和自身
    # len(query_sets) = 23 # oxford
    # len(query_sets) = 5  # others
    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            # 除去自身
            if i == j:
                continue
            # pair_recall np.array 是(25,)的数组 存avg recall @N (1~25)
            # pair_similarity list 是TP情况下的top1相似度
            # pair_opr是 recall @1% 
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats


def load_pc(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc


def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    # 验证的旋转变换
    valtrans = ValTransform(1)

    model.eval()
    embeddings_l = []
    for elem_ndx in set:
        x = load_pc(set[elem_ndx]["query"], params)

        # 尝试在这里对x进行旋转
        # x = valtrans(x)

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            coords = ME.utils.sparse_quantize(coordinates=x,
                                              quantization_size=params.model_params.mink_quantization_size)
            bcoords = ME.utils.batched_coordinates([coords])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

            embedding, weight = model(batch)
            # embedding is (1, 1024) tensor?
            # 应该是print(embedding.shape) --> torch.Size([1, 256])
            # (1, feature_size)
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    """
    # 计算recall@%1
    """
    # Original PointNetVLAD code
    # database和query分别来自m, n采集序列
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    
    # print("database_output queries_output", len(database_output), len(queries_output))
    # database_output queries_output 440 120
    # 440 120 代表子图的个数

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    # 特征上的KDTree
    database_nbrs = KDTree(database_output)

    # 相邻的数量？
    num_neighbors = 25
    # recall 记录特征最近的前25个子图的recall
    recall = [0] * num_neighbors

    # top%1的相似分数
    top1_similarity_score = []
    # top%1的检索数量
    one_percent_retrieved = 0
    # 阈值 = database的子图数量除以100取整 代表了top%1的数量
    threshold = max(int(round(len(database_output)/100.0)), 1)

    # 评估的数量
    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': , 0:[], 1:[]}
        # 真正在m中的邻居 记录于生成数据的时候
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        # 找与queries_output[i]最近的25个database的索引
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        # indices[0]是最近的25个索引值
        for j in range(len(indices[0])):
            # j在true_neighbors里面
            if indices[0][j] in true_neighbors:
                # j == 0 是最接近的那个子图 记为top1_similarity_score
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                # 一旦找到一个nbr则退出
                break
        # intersection 将两个集合的交集作为新集合返回
        # 取indices的前threshold个子图索引与真实的匹配的true_neighbors作交集
        # 交集不为空则视为recall@%1成功！
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    # print("num_evaluated", num_evaluated)
    # print("one_percent_retrieved", one_percent_retrieved)
    # print("recall", recall)
    # num_evaluated 150
    # one_percent_retrieved 147
    # recall [138, 6, 3, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    # 累积和 cumsum: Return the cumulative sum of the elements along a given axis.
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, silent=False)
    print_eval_stats(stats)
