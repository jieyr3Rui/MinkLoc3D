# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from matplotlib.cbook import contiguous_regions
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import tqdm
import MinkowskiEngine as ME
import random
import scipy

from misc.utils import MinkLocParams
from models.model_factory import model_factory

# 加载矫正transformer
from models.transformer import transformer

import matplotlib.pyplot as plt

# 导入ValTransform 
from datasets.oxford import ValTransform

def evaluate(model, modelCorrect, device, params, silent=True):
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

        temp = evaluate_dataset(model, modelCorrect, device, params, database_sets, query_sets, silent=silent)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, modelCorrect, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    modelCorrect.eval()

    for set in tqdm.tqdm(database_sets, disable=silent):
        database_embeddings.append(get_latent_vectors(model, modelCorrect, set, device, params))

    for set in tqdm.tqdm(query_sets, disable=silent):
        query_embeddings.append(get_latent_vectors(model, modelCorrect, set, device, params))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j:
                continue
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


def getMFormAxisAndTheta(axis, theta):
    """
    # so3计算axis和theta下的旋转矩阵\n
    axis为3维向量，取其归一化值，再乘以旋转角度theta，因此axis / norm(axis) * theta为一个旋转向量。\n
    numpy.cross 返回两个（数组）向量的叉积。\n
    expm 使用Pade近似计算矩阵指数。\n
    """
    return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta)).astype(np.float32)

def correctPointCloudWithZ(pc:np.ndarray, currZ:np.ndarray):
    """
    # 矫正点云
    """
    currZ = np.array(currZ / np.linalg.norm(currZ), dtype=np.float32)
    realz = np.array([0,0,1], dtype=np.float32)
    roAxis = np.cross(currZ, realz)
    roTheta = np.arccos(np.sum(currZ * realz))
    # print(roAxis, roTheta)
    
    roMatrix = getMFormAxisAndTheta(axis=roAxis, theta=roTheta)
    # print(roMatrix)
    pc = np.dot(roMatrix, pc)

    return pc


def zCorrect(modelCorrect, pc:torch.Tensor, device):
    """
    # 矫正点云z的角度 随即旋转 so3 --> z

    """
    assert pc.size()[1] == 3, "Error pc size!"
    pc = pc.unsqueeze(dim=0).transpose(1,2).to(device)
    predz = modelCorrect(pc)
    predz = np.array(predz.squeeze().cpu(), dtype=np.float32)
    pc = np.array(pc.squeeze().cpu(), dtype=np.float32)
    # print(pc)
    pc = correctPointCloudWithZ(pc, predz)
    # print(pc)
    # contiguous保证内存连续性
    pc = torch.tensor(pc, dtype=torch.float).transpose(0,1).cpu().contiguous()

    return pc

def get_latent_vectors(model, modelCorrect, set, device, params):
    # Adapted from original PointNetVLAD code

    # 验证的旋转变换
    valtrans = ValTransform(1)

    model.eval()
    modelCorrect.eval()

    embeddings_l = []
    for elem_ndx in set:
        x = load_pc(set[elem_ndx]["query"], params)

        # 尝试在这里对x进行旋转
        x = valtrans(x)

    
        with torch.no_grad():

            x = zCorrect(modelCorrect, x, device)


            # coords are (n_clouds, num_points, channels) tensor
            coords = ME.utils.sparse_quantize(coordinates=x,
                                              quantization_size=params.model_params.mink_quantization_size)
            
            bcoords = ME.utils.batched_coordinates([coords])
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

            embedding = model(batch)
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
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
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


    # 加载矫正tranformer
    weightPath = "/home/jieyr/code/MinkLoc3D/weights"
    weightName = "transformer_best.pth"
    modelCorrect = transformer(points_num=4096)
    # 加载模型参数
    weights = os.path.join(weightPath, weightName)
    if weights is not None:
        assert os.path.exists(weights), 'Cannot open network weights: {}'.format(weights)
        print('Loading weightsCorrect: {}'.format(weights))
        modelCorrect.load_state_dict(torch.load(weights, map_location=device))
    # 转到cuda
    modelCorrect.to(device)


    stats = evaluate(model, modelCorrect, device, params, silent=False)
    print_eval_stats(stats)
