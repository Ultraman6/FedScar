import copy, os
import torch
import argparse
import warnings
import numpy as np
import logging
import random
import itertools
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serverinit import FedInit
from flcore.servers.serverscaffold import Scaffold
from flcore.servers.serversam import FedSAM
from flcore.servers.servergamma import FedGamma
from flcore.servers.serversmoo import FedSMOO
from flcore.servers.serverlesam import FedLESAM
from flcore.servers.serverlesams import FedLESAMS
from flcore.servers.serverlesamd import FedLESAMD
from flcore.servers.servergmt import FedGMT
from flcore.servers.servergmt_v2 import FedGMTV2
from flcore.servers.serverscar import FedScar
from flcore.servers.serverscarle import FedScarle
from flcore.servers.serverscars import FedScars
from flcore.servers.serverscarl import FedScarl
from flcore.servers.serverscar1 import FedScar1
from flcore.servers.serverscar2 import FedScar2
from flcore.servers.serverscar3 import FedScar3
from flcore.servers.serverscar4 import FedScar4
from flcore.trainmodel.resnet import resnet8, resnet18
from flcore.trainmodel.CNN import FedAvgNetCIFAR
from flcore.trainmodel.vit import ViT
from flcore.trainmodel.nlp import fastText
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def reset_random_seeds(seed):
    # 设置随机种子
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def expand_configs(algorithm_configs):
    """
    展开算法配置，将列表超参数展开成多个任务
    例如: {'rho': [0.1, 0.2], 'beta': [0.1]} -> 
         [{'rho': 0.1, 'beta': 0.1}, {'rho': 0.2, 'beta': 0.1}]
    """
    expanded_tasks = []
    for algo_class, params, timestr in algorithm_configs:
        # 分离列表参数和单值参数
        list_params = {}
        single_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                list_params[key] = value
            else:
                single_params[key] = value
        
        # 如果有列表参数，做笛卡尔积展开
        if list_params:
            # 获取所有列表参数的键
            list_keys = list(list_params.keys())
            list_values = list(list_params.values())
            
            # 生成笛卡尔积
            for combination in itertools.product(*list_values):
                task_params = single_params.copy()
                for key, value in zip(list_keys, combination):
                    task_params[key] = value
                expanded_tasks.append((algo_class, task_params, timestr))
        else:
            # 没有列表参数，直接添加
            expanded_tasks.append((algo_class, single_params, timestr))
    
    return expanded_tasks

def run_single_task(base_args, algorithm_class, params, timestr):
    """
    执行单个任务
    Args:
        base_args: 基础参数对象
        algorithm_class: 算法类（例如 FedGMT）
        params: 算法超参数字典
    """
    # 复制基础参数
    args = copy.deepcopy(base_args)
    # 从算法类获取算法名称
    args.algorithm = algorithm_class.__name__
    reset_random_seeds(args.seed)
    # 创建模型
    print(f"\n{'='*60}")
    print(f"Algorithm: {args.algorithm} Params: {params}")
    print(f"{'='*60}")
    
    if args.dataset == 'cinic10':
        args.model = ViT().to(args.device)
    elif args.dataset == 'cifar10':
        args.model = FedAvgNetCIFAR().to(args.device)
    elif args.dataset == 'cifar100':
        args.model = resnet8().to(args.device)
    elif args.dataset == 'agnews':
        args.model = fastText().to(args.device, num_classes=4)
    elif args.dataset == 'sogounews':
        args.model = fastText().to(args.device, num_classes=5)
    else:
        print("check args.dataset !!!!!!")
        return
    
    # 创建服务器并训练（直接使用算法类初始化）
    reporter = MemReporter()
    print("Creating server and clients ...")
    
    try:
        server = algorithm_class(args, hyperparams=params)
        if args.cal_sharp and timestr:
            server.sharpness_v2(timestr, args.round_list)
        elif args.loss_land and timestr:
            server.visualize_loss_landscape(timestr, args.round_list)
        else:
            server._save_source_files()
            server.train()
        print(f"\nTask completed: {args.algorithm} with params {params}")
    except Exception as e:
        print(f"\nError in task {args.algorithm} with params {params}: {e}")
        import traceback
        traceback.print_exc()
    
    reporter.report()

def run(base_args, algorithm_configs):
    """
    执行所有配置的算法任务
    Args:
        base_args: 基础参数对象
        algorithm_configs: 算法配置列表，格式: [(算法类, {超参数字典}), ...]
    """
    # 展开配置
    tasks = expand_configs(algorithm_configs)
    
    print(f"\n{'='*60}")
    print(f"Total tasks to execute: {len(tasks)}")
    print(f"{'='*60}\n")
    
    # 执行每个任务
    for idx, (algorithm_class, params, timestr) in enumerate(tasks, 1):
        print(f"\n{'#'*60}")
        print(f"Task {idx}/{len(tasks)}")
        print(f"{'#'*60}")
        run_single_task(base_args, algorithm_class, params, timestr)
    
    print(f"\n{'='*60}")
    print("All tasks completed!")
    print(f"{'='*60}\n")

def load_data_partition_config(data_partition_dir):
    """
    从数据划分目录加载配置文件
    Args:
        data_partition_dir: 数据划分目录路径（包含config.json的目录）
    Returns:
        config字典，如果加载失败返回None
    """
    try:
        import ujson
    except ImportError:
        import json as ujson
    
    config_path = os.path.join(data_partition_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        return config
    except Exception as e:
        print(f"Error: Failed to load config from {config_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_args_from_data_partition(args, data_partition_dir):
    """
    从数据划分目录的config.json设置args参数
    Args:
        args: 参数对象
        data_partition_dir: 数据划分目录路径（相对路径或绝对路径）
    Returns:
        更新后的args对象
    """
    # 规范化路径：如果是相对路径，转换为绝对路径
    if not os.path.isabs(data_partition_dir):
        # 相对路径：相对于当前工作目录
        data_partition_dir = os.path.abspath(data_partition_dir)
    else:
        # 已经是绝对路径，确保规范化
        data_partition_dir = os.path.normpath(data_partition_dir)
    
    # 验证目录是否存在
    if not os.path.isdir(data_partition_dir):
        print(f"Error: Data partition directory does not exist: {data_partition_dir}")
        return None
    
    # 验证关键文件是否存在
    config_path = os.path.join(data_partition_dir, 'config.json')
    train_dir = os.path.join(data_partition_dir, 'train')
    test_dir = os.path.join(data_partition_dir, 'test')
    
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {data_partition_dir}")
        return None
    if not os.path.isdir(train_dir):
        print(f"Warning: train/ directory not found in {data_partition_dir}")
    if not os.path.isdir(test_dir):
        print(f"Warning: test/ directory not found in {data_partition_dir}")
    
    config = load_data_partition_config(data_partition_dir)
    if config is None:
        return None
    
    # 设置数据划分目录（使用绝对路径）
    args.data_partition_dir = data_partition_dir
    
    # 从config读取并设置参数（优先使用config中的值）
    if 'num_clients' in config:
        args.num_clients = config['num_clients']
    if 'num_classes' in config:
        args.num_classes = config['num_classes']
    if 'seed' in config and args.seed == 1:  # 只有在使用默认seed时才覆盖
        args.seed = config['seed']
    
    # 从目录名推断数据集名称（如果未通过命令行参数指定）
    # 例如: data/cifar10/seed1_nc100_LDA_alpha0.1_imb0.5 -> cifar10
    if args.dataset is None:
        # 尝试从路径推断
        # os.path.normpath 已经处理了路径分隔符
        normalized_path = os.path.normpath(data_partition_dir)
        path_parts = normalized_path.split(os.sep)
        for part in path_parts:
            if part in ['cifar10', 'cifar100', 'cinic10', 'fminist', 'agnews']:
                args.dataset = part
                break
    
    # 如果仍然无法确定数据集，使用默认值
    if args.dataset is None:
        print(f"Warning: Cannot infer dataset name from path {data_partition_dir}, using default 'cifar10'")
        args.dataset = 'cifar10'
    
    # 设置数据集类别数（如果config中没有）
    if not hasattr(args, 'num_classes') or args.num_classes is None:
        if args.dataset == 'cinic10':
            args.num_classes = 10
        elif args.dataset == 'cifar10':
            args.num_classes = 10
        elif args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset == 'agnews':
            args.num_classes = 4
        elif args.dataset == 'fminist':
            args.num_classes = 10
        else:
            print(f"Warning: Unknown dataset {args.dataset}, using default num_classes=10")
            args.num_classes = 10
    
    print(f"\n{'='*60}")
    print(f"Data Partition Configuration:")
    print(f"{'='*60}")
    print(f"Data partition directory: {args.data_partition_dir}")
    print(f"  - num_clients: {args.num_clients}")
    print(f"  - num_classes: {args.num_classes}")
    print(f"  - dataset: {args.dataset}")
    if 'data_partition' in config:
        print(f"  - data_partition: {config['data_partition']}")
    if 'seed' in config:
        print(f"  - seed: {args.seed}")
    print(f"{'='*60}\n")
    
    return args

ALGORITHM_CONFIGS = [
    # (FedAvg, {}, '2512051059'),
    # (FedDyn, {'alpha': 0.01}, ''),
    # (FedInit, {'beta': 0.1}, ''),
    # (Scaffold, {}, ''),
    # (FedSAM, {'rho': [0.25]}, '2512051710'),
    # (FedGamma, {'rho': [0.25]}, '2601151339'),
    # (FedSMOO, {'rho': [0.25], 'beta': 0.1}, '2601151402'),
    # (FedLESAM, {'rho': [0.25]}, '2601151440'),
    # (FedLESAMS, {'rho': [0.2, 1.0, 1.5, 2.0, 3.0]}, '2601151440'),
    # (FedLESAMD, {'rho': [3.0], 'beta': 0.1}, '2601151440'),
    (FedGMT, {'alpha': 0.95, 'beta': 0.01, 'gama': 1.0, 'tau': 3.0}, '2601151501'),
    # (FedScar, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.06, 'rho': [0.1]}, '2511220058'),
    # (FedScarle, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.1, 'rho': [0.2, 1.0, 1.5, 2.0, 3.0]}, '2511220058'),
    # (FedGMTV2, {'alpha': 0.5, 'beta': 0.1, 'gama': 1.0, 'tau': 3.0}),
    # (FedScar, {'alpha': 0.01, 'gama': 0.9, 'beta': 0.01, 'rho': 0.0}, '2601151547'),
    # (FedScars, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.06, 'rho': 0.0}, ''),
    # (FedScarl, {'alpha': 0.1, 'gama': 0.9, 'tau': 0.1, 'rho': 0.0}, ''),
    # (FedScar1, {'alpha': 0.1, 'gama': 0.9, 'rho': 0.0}),
    # (FedScar2, {'beta': 0.06, 'rho': 0.0}),
    # (FedScar3, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.06, 'rho': 0.0}),
    # (FedScar4, {'alpha': 0.01, 'beta': 0.1},''),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rl', "--round_list", nargs='+', type=int, default=[500])
    parser.add_argument('-cs', "--cal_sharp", type=bool, default=False)
    parser.add_argument('-ll', "--loss_land", type=bool, default=False)
    parser.add_argument('-sm', "--save_model", type=bool, default=False)
    parser.add_argument('-sl', "--save_local", type=bool, default=False)
    parser.add_argument('-dev', "--device", type=str, default="cuda:0")
    parser.add_argument('-data', "--dataset", type=str, default=None)
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-le', "--local_epochs", nargs='+', type=int, default=[5])
    parser.add_argument('-lbs', "--batch_size", nargs='+', type=int, default=[50])
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument("--save_gap", type=int, default=100)
    parser.add_argument("--eval_round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('-jr', "--join_ratio", type=float, nargs='+', default=[0.1])
    parser.add_argument('--data_partition', type=str, nargs='+', default=[
        # 'cifar10/seed1_nc100_LDA_alpha1_imb0.5',
        # 'cifar10/seed1_nc100_LDA_alpha0.1_imb0.5',
        'cifar10/seed1_nc100_LDA_alpha0.01_imb0.5',
        # 'cifar10/seed1_nc100_shard_shard10',
        # 'cifar10/seed1_nc100_shard_shard2',
        # 'cifar100/seed1_nc100_LDA_alpha1_imb0.5',
        # 'cifar100/seed1_nc100_LDA_alpha0.1_imb0.5',
        # 'cifar100/seed1_nc200_LDA_alpha0.1_imb0.5',
        # 'cifar100/seed1_nc100_LDA_alpha0.01_imb0.5',
        # 'cifar100/seed1_nc100_shard_shard20',
        # 'cifar100/seed1_nc100_shard_shard10',
        # 'cinic10/seed1_nc100_LDA_alpha0.1_imb0.5',
        # 'cinic10/seed1_nc100_shard_shard2',
        # 'agnews/seed1_nc100_LDA_alpha0.1_imb0.5',
        # 'agnews/seed1_nc100_shard_shard2',
    ])
    args = parser.parse_args()
    join_ratios = args.join_ratio
    data_partitions = args.data_partition
    local_epochs = args.local_epochs
    batch_sizes = args.batch_size
    for batch_size, local_epoch, join_ratio, data_partition in itertools.product(batch_sizes, local_epochs, join_ratios, data_partitions):
        args.batch_size = batch_size
        args.local_epochs = local_epoch
        args.join_ratio = join_ratio
        args.data_partition_dir = os.path.join('/mnt/c/Github/FL-SAM/dataset/data', data_partition)
        # 如果指定了数据划分目录，从config.json加载配置
        if args.data_partition_dir:
            args = setup_args_from_data_partition(args, args.data_partition_dir)
            if args is None:
                print("Failed to load data partition config. Exiting...")
                exit(1)
        else:
            # 如果没有指定数据划分目录，使用默认路径和参数
            args.data_partition_dir = None
            # 设置数据集类别数
            if args.dataset is None:
                args.dataset = 'cifar10'
            if args.num_clients is None:
                args.num_clients = 100
            if args.dataset in ['cinic10', 'cifar10', 'fminist']:
                args.num_classes = 10
            elif args.dataset in ['cifar100']:
                args.num_classes = 100
            elif args.dataset in ['agnews']:
                args.num_classes = 4
            else:
                print("check your dataset!!!!!!!!!!!!!")
                exit()

        # 打印基础配置信息
        print("=" * 60)
        print("Base Configuration:")
        print("=" * 60)
        print("Local batch size: {}".format(args.batch_size))
        print("Local steps: {}".format(args.local_epochs))
        print("Local learning rate: {}".format(args.local_learning_rate))
        print("Total number of clients: {}".format(args.num_clients))
        print("Clients join in each round: {}".format(args.join_ratio))
        print("Dataset: {}".format(args.dataset))
        print("Number of classes: {}".format(args.num_classes))
        print("Using device: {}".format(args.device))
        print("Seed: {}".format(args.seed))
        if args.data_partition_dir:
            print("Data partition directory: {}".format(args.data_partition_dir))
        print("=" * 60)
        print("\nAlgorithm configurations will be loaded from ALGORITHM_CONFIGS")
        print("=" * 60)

        # 执行算法任务
        run(args, ALGORITHM_CONFIGS)



