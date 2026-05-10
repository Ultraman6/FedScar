
import os
import torch
import ujson
from torch.utils.data.dataset import Dataset
import numpy as np

# 全局变量存储数据划分目录（如果指定了的话）
_DATA_PARTITION_DIR = None

def set_data_partition_dir(data_partition_dir):
    """
    设置数据划分目录
    Args:
        data_partition_dir: 数据划分目录路径（包含config.json, train/, test/的目录）
    """
    global _DATA_PARTITION_DIR
    _DATA_PARTITION_DIR = data_partition_dir


def get_data_dir(dataset):
    """
    获取数据目录路径
    Args:
        dataset: 数据集名称
    Returns:
        数据目录路径
    """
    global _DATA_PARTITION_DIR
    if _DATA_PARTITION_DIR is not None:
        # 使用指定的数据划分目录
        return _DATA_PARTITION_DIR
    else:
        # 使用默认路径
        return os.path.join('../dataset/data', dataset)


def read_data(dataset, idx, is_train=True):
    """
    读取数据文件
    Args:
        dataset: 数据集名称
        idx: 客户端索引（训练数据）或None（测试数据）
        is_train: 是否为训练数据
    Returns:
        数据
    """
    data_dir = get_data_dir(dataset)
    
    if is_train:
        train_data_dir = os.path.join(data_dir, 'train')
        train_file = os.path.join(train_data_dir, str(idx) + '.pkl')
        train_data = torch.load(train_file)
        return train_data
    else:
        test_data_dir = os.path.join(data_dir, 'test')
        test_file = os.path.join(test_data_dir, 'test.pkl')
        test_data = torch.load(test_file)
        return test_data


def read_client_data(dataset, idx):
    train_data = read_data(dataset, idx)
    return train_data

def read_client_json(dataset, idx):
    """
    读取客户端JSON配置信息
    Args:
        dataset: 数据集名称
        idx: 客户端索引
    Returns:
        客户端数据分布信息
    """
    data_dir = get_data_dir(dataset)
    json_file = os.path.join(data_dir, 'config.json')
    with open(json_file, 'r') as f:
        config = ujson.load(f)
    trainsamples = config['Size of samples for labels in clients'][idx]
    return trainsamples

def read_total_json(dataset):
    """
    读取总体JSON配置信息
    Args:
        dataset: 数据集名称
    Returns:
        总体类别数量信息
    """
    data_dir = get_data_dir(dataset)
    json_file = os.path.join(data_dir, 'config.json')
    with open(json_file, 'r') as f:
        config = ujson.load(f)
    total = config['Total class number']
    return total

from torch.utils.data import DataLoader
def class_dataloader_for_MAS(dataset, idx,batch_size):
    client_data = read_client_data(dataset, idx)
    client_dict_per_class= read_client_json(dataset, idx)[1]
    client_data_num = []
    for i in client_dict_per_class:
        client_data_num.append(float(i[1]))

    #client_data_num = [x for x in client_data_num if x != 0]

    class_client_data = {}
    for i in range(len(client_data_num)):
        class_client_data[i] = []
    for input,label in client_data:
        class_client_data[label].append([input,label])

    dataloader = []
    for i in range(len(client_data_num)):  
        if len(class_client_data[i])!=0:
            dataloader.append(DataLoader(class_client_data[i], batch_size,pin_memory=True))

    client_data_num = [x for x in client_data_num if x != 0]
    # print(client_data_num)
    client_data_num = torch.tensor(client_data_num)
    client_data_num = client_data_num/sum(client_data_num)
    client_data_weight= -client_data_num.log()

 
    return dataloader,client_data_weight




class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def repair_cov(matrix, factor=0.001):
    matrix = torch.tensor(matrix).cuda()
    matrix = (matrix+matrix.T)/2
    w, v = torch.linalg.eig(matrix)
    w = w.real
    v= v.real
    if torch.all(w >= factor):
        m = matrix
    else:
        w[w < factor] = factor
        m = torch.matmul(torch.matmul(v, torch.diag(w)), v.T)
    
    return m


def get_head_class(dataset,head_ratio):
    total = read_total_json(dataset)
    sum = 0
    for t in total:
        sum+=t[1]
    cut = 0
    for t in total:
        cut+=t[1]
        if cut>=sum*head_ratio:
            return t[0]+1

def remix(image, label,num_class_list,device):
    r"""
    Reference:
        Chou et al. Remix: Rebalanced Mixup, ECCV 2020 workshop.
    The difference between input mixup and remix is that remix assigns lambdas of mixed labels
    according to the number of images of each class.
    Args:
        tau (float or double): a hyper-parameter
        kappa (float or double): a hyper-parameter
        See Equation (10) in original paper (https://arxiv.org/pdf/2007.03943.pdf) for more details.
    """
    assert num_class_list is not None, "num_class_list is required"
    class_num_list = []
    for i in num_class_list:
        class_num_list.append(i[1])
    num_class_list = torch.FloatTensor(class_num_list).to(device)
    alpha = 1
    remix_tau = 0.5
    remix_kappa = 3
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0))
    image_a, image_b = image, image[idx]
    label_a, label_b = label, label[idx]
    mixed_image = l * image_a + (1 - l) * image_b
    mixed_image = mixed_image.to(device)

    #what remix does
    l_list = torch.empty(image.shape[0]).fill_(l).float().to(device)
    n_i, n_j = num_class_list[label_a], num_class_list[label_b].float()

    if l < remix_tau:
        l_list[n_i/n_j >= remix_kappa] = 0
    if 1 - l < remix_tau:
        l_list[(n_i*remix_kappa)/n_j <= 1] = 1

    label_a = label_a.to(device)
    label_b = label_b.to(device)
    # loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
    # loss = loss.mean()

    return mixed_image,label_a,label_b,l_list

import random
from PIL import Image

class CIFARDecorator(Dataset):
    """
    A decorator class that wraps existing CIFAR datasets (or lists) and adds:
    - Noise injection
    - Additional augmentations (like Cutout)
    - Support for different modes (all, labeled, unlabeled, test)
    """

    def __init__(self, base_dataset,
                 noise_ratio=0.0, noise_mode='sym', mode='all',
                 transform=None, target_transform=None, noise_file=None,
                 pred=None, probability=None):

        self.base_dataset = base_dataset
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.mode = mode

        # Handle transforms
        self.transform = transform if transform is not None else getattr(base_dataset, 'transform', None)
        self.target_transform = target_transform if target_transform is not None else getattr(base_dataset,
                                                                                              'target_transform', None)

        # -----------------------------------------------------------
        # Data & Label Extraction Logic (Modified for list support)
        # -----------------------------------------------------------
        self.data = []
        self.original_labels = []

        # Case 1: base_dataset is a simple python list [(img, label), ...]
        if isinstance(base_dataset, list):
            # Unzip the list efficiently
            inputs, labels = zip(*base_dataset)

            # Ensure data is a numpy array for indexing (consistent with CIFAR objects)
            # Converting elements to np.array first ensures PIL images/Tensors are handled
            self.data = np.array([np.array(x) for x in inputs])
            self.original_labels = list(labels)

            # Infer num_classes from data since 'classes' attr is missing
            self.num_classes = len(set(self.original_labels))

        # Case 2: Standard torchvision CIFAR object (Fast path)
        elif hasattr(base_dataset, 'data') and (hasattr(base_dataset, 'targets') or hasattr(base_dataset, 'labels')):
            self.data = base_dataset.data
            if hasattr(base_dataset, 'targets'):
                self.original_labels = base_dataset.targets
            else:
                self.original_labels = base_dataset.labels

            if hasattr(base_dataset, 'classes'):
                self.num_classes = len(base_dataset.classes)
            else:
                self.num_classes = 100 if 'CIFAR100' in str(type(base_dataset)) else 10

        # Case 3: Generic Dataset object (Slow path - fallback)
        else:
            for i in range(len(base_dataset)):
                img, label = base_dataset[i]
                self.data.append(np.array(img))
                self.original_labels.append(label)
            self.data = np.array(self.data)

            # Try to infer classes, fallback to assumption
            if hasattr(base_dataset, 'classes'):
                self.num_classes = len(base_dataset.classes)
            else:
                unique_labels = set(self.original_labels)
                self.num_classes = len(unique_labels) if unique_labels else 10
        # -----------------------------------------------------------

        # Handle noise injection
        if self.noise_ratio > 0 and mode != 'test':
            if noise_file and os.path.exists(noise_file):
                with open(noise_file, 'r') as f:
                    self.noise_labels = json.load(f)
            else:
                # Assuming NoiseInjector is defined elsewhere
                noise_injector = NoiseInjector(noise_ratio, noise_mode, self.num_classes)
                self.noise_labels = noise_injector.inject_noise(self.original_labels)

                if noise_file:
                    with open(noise_file, 'w') as f:
                        json.dump(self.noise_labels, f)
        else:
            self.noise_labels = self.original_labels

        # Handle different modes (slicing data)
        if mode in ['labeled', 'unlabeled'] and pred is not None:
            if mode == 'labeled':
                self.indices = pred.nonzero()[0]
                self.probability = [probability[i] for i in self.indices] if probability is not None else None
            else:  # unlabeled
                self.indices = (1 - pred).nonzero()[0]
                self.probability = None

            # Filter data and labels
            self.data = self.data[self.indices]
            self.noise_labels = [self.noise_labels[i] for i in self.indices]
        else:
            self.indices = None
            self.probability = None

        # Explicitly set targets for compatibility
        self.targets = self.noise_labels

    def _apply_target_transform(self, target):
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

    def _process_image(self, img):
        if not isinstance(img, Image.Image):
            # Handle numpy arrays: ensure usually (H, W, C) for PIL
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    # Normal RGB image
                    img = Image.fromarray(img.astype('uint8'), 'RGB')
                elif img.ndim == 3 and img.shape[0] == 3:
                    # Handle (C, H, W) case just in case, though standard CIFAR is HWC
                    img = np.transpose(img, (1, 2, 0))
                    img = Image.fromarray(img.astype('uint8'), 'RGB')
                else:
                    # Grayscale or flattened
                    img = Image.fromarray(img.astype('uint8'))
            else:
                # Fallback
                img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.noise_labels[index]

        img = self._process_image(img)

        if self.mode == 'labeled' and self.probability is not None:
            img2 = self._process_image(self.data[index])
            transformed_target = self._apply_target_transform(target)
            return img, img2, transformed_target, self.probability[index]

        elif self.mode == 'unlabeled':
            img2 = self._process_image(self.data[index])
            return img, img2

        else:
            transformed_target = self._apply_target_transform(target)
            return img, transformed_target

class NoiseInjector:
    """Handles noise injection for labels"""

    def __init__(self, noise_ratio=0.4, noise_mode='sym', num_classes=10):
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.num_classes = num_classes
        # Class transition for asymmetric noise (CIFAR10 specific)
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

    def inject_noise(self, labels):
        """Inject noise into labels"""
        num_samples = len(labels)
        noise_labels = []

        # Randomly select samples to corrupt
        idx = list(range(num_samples))
        random.shuffle(idx)
        num_noise = int(self.noise_ratio * num_samples)
        noise_idx = idx[:num_noise]

        for i in range(num_samples):
            if i in noise_idx:
                if self.noise_mode == 'sym':
                    # Symmetric noise: randomly change to any class
                    noise_label = random.randint(0, self.num_classes - 1)
                    noise_labels.append(noise_label)
                elif self.noise_mode == 'asym':
                    # Asymmetric noise: use predefined transitions
                    if self.num_classes == 10:  # CIFAR10
                        noise_label = self.transition.get(labels[i], labels[i])
                    else:
                        # For other datasets, fallback to symmetric
                        noise_label = random.randint(0, self.num_classes - 1)
                    noise_labels.append(noise_label)
            else:
                # Keep original label
                noise_labels.append(labels[i])

        return noise_labels