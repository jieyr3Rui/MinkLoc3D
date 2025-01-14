# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset, TrainTransform, TrainSetTransform, ValTransform
from datasets.samplers import BatchSampler
from misc.utils import MinkLocParams


def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)
    # MinkLoc3D代码的BUG!!!!!!
    # train_transform不行，传进去不知道是哪个变量
    # transform=train_transform 要这样
    datasets['train'] = OxfordDataset(params.dataset_folder, params.train_file, transform=train_transform,
                                      set_transform=train_set_transform)

    val_transform = ValTransform(params.aug_mode)
    # val_transform = None
    if params.val_file is not None:
        datasets['val'] = OxfordDataset(params.dataset_folder, params.val_file, transform=val_transform)
    return datasets


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    # 自定义dataloder的输出，data_list是OxfordDataset.__getitem__的样本组成的list
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)

        if mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch = {'cloud': batch}
        else:
            # Given coordinates, and features (optionally labels), the function generates quantized (voxelized) coordinates.
            # 给定坐标系和坐标，sparse_quantize生成量化体素化坐标，batch:(batch_size, n_points, 3)
            # 这里配置文件中的mink_quantization_size=0.01，理解为每个体素0.01m^3
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                      for e in batch]
            # Create a ME.SparseTensor coordinates from a sequence of coordinates
            # Given a list of either numpy or pytorch tensor coordinates, 
            # return the batched coordinates suitable for ME.SparseTensor.
            # coords: [[elemetID, x, y, z]]，这个时候把所有的样本都集中在一个维度
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            # Assign a dummy feature equal to 1 to each point 
            # 为每个点指定一个等于1的虚拟特征
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            # feats.shape = (batch_size*n_points, 1)
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            # 新的batch组织形式
            batch = {'coords': coords, 'features': feats}

        # Compute positives and negatives mask
        # Compute positives and negatives mask
        # mask是二维数组，bs x bs的01矩阵，反映的是一个batch里面各个样本的positive跟negative的关系
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        # 不在non_negatives的就是negatives 
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  params.model_params.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], params.model_params.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
