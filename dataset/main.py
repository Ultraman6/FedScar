import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from utils.dataset_utils import *
from utils.long_tailed_cifar import train_long_tail
from utils.sample_dirichlet import clients_indices
import copy
import argparse
from tqdm import tqdm

def generate_dir_name(num_clients, seed, data_partition, niid=None, imb=None, shard_per_user=None):
    """
    Generate directory name based on dataset partition parameters.
    
    Args:
        num_clients: number of clients
        seed: random seed
        data_partition: 'LDA' or 'shard'
        niid: non-IID alpha parameter (for LDA)
        imb: imbalance factor
        shard_per_user: shard per user (for shard)
    
    Returns:
        Directory name string, e.g., "seed1_nc100_LDA_alpha0.1_imb0.5" or "seed1_nc100_shard_shard2"
    """
    dir_name = f"seed{seed}_nc{num_clients}_{data_partition}"
    
    if data_partition == 'LDA':
        if niid is not None:
            # Format alpha to avoid too many decimal places
            # Handle both integer and float values
            if isinstance(niid, float):
                alpha_str = f"{niid:.6f}".rstrip('0').rstrip('.')
            else:
                alpha_str = str(niid)
            dir_name += f"_alpha{alpha_str}"
        if imb is not None:
            # Format imb to avoid too many decimal places
            if isinstance(imb, float):
                imb_str = f"{imb:.6f}".rstrip('0').rstrip('.')
            else:
                imb_str = str(imb)
            dir_name += f"_imb{imb_str}"
    elif data_partition == 'shard':
        if shard_per_user is not None:
            dir_name += f"_shard{shard_per_user}"
    return dir_name

# Allocate data to users
def generate_data(dir_path, num_clients, num_classes, niid, imb,seed,args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Determine data partition method
    data_partition = None
    if args.LDA and not args.shard:
        data_partition = 'LDA'
    elif not args.LDA and args.shard:
        data_partition = 'shard'
    else:
        print("check LDA and shard input!!!!")
        exit()
    
    # Generate subdirectory name based on parameters
    subdir_name = generate_dir_name(num_clients, seed, data_partition, niid, imb, args.shard_per_user)
    partition_dir = os.path.join(dir_path, subdir_name)
    
    # Setup directory for train/test data
    config_path = os.path.join(partition_dir, "config.json")
    train_path = os.path.join(partition_dir, "train")
    test_path = os.path.join(partition_dir, "test")
    
    # Check if dataset already exists before doing any work
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, imb, seed, args.shard_per_user, args):
        print("Skipping data generation.\n")
        return
    
    # Dataset doesn't exist, proceed with generation
    # Create directories if they don't exist
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    train_path += os.sep  # Add trailing separator for consistency
    test_path += os.sep
    
    print(f"Generating dataset at: {partition_dir}")
    
    if args.dataset == 'cifar10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        # Get Cifar10 data
        transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
        [transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])

        trainset = torchvision.datasets.CIFAR100(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)
    elif args.dataset == 'fminist':
        transform = transforms.Compose(
        [transforms.ToTensor(), 
            transforms.Normalize((0.2860366729433025), (0.35288708155778725))
            ])

        trainset = torchvision.datasets.FashionMNIST(
            root=dir_path+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(
            root=dir_path+"rawdata", train=False, download=True, transform=transform)

    elif args.dataset == 'cinic10':
        cinic_directory = dir_path+"rawdata"
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        trainset = torchvision.datasets.ImageFolder(
            cinic_directory + '/train', transform=transform)
        testset = torchvision.datasets.ImageFolder(
            cinic_directory + '/test', transform=transform)

    elif args.dataset == 'tinyimagenet':
        # Tiny-ImageNet: 200 类，每类 500 训练图、50 验证图，64x64 RGB
        tiny_mean = (0.5, 0.5, 0.5)
        tiny_std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std),
        ])

        raw_root = dir_path + "rawdata"
        tiny_root = os.path.join(raw_root, "tiny-imagenet-200")
        if not os.path.exists(tiny_root):
            os.makedirs(raw_root, exist_ok=True)
            os.system(f"wget --directory-prefix {raw_root}/ http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            os.system(f"unzip {raw_root}/tiny-imagenet-200.zip -d {raw_root}/")

        trainset = torchvision.datasets.ImageFolder(
            os.path.join(tiny_root, "train"), transform=transform
        )
        # Tiny-ImageNet 官方划分的 val 在 val/ 下，需要按 annotations 映射标签
        val_dir = os.path.join(tiny_root, "val")
        val_annot = os.path.join(val_dir, "val_annotations.txt")
        if os.path.exists(val_annot):
            # 按官方标注重建 val 目录结构再用 ImageFolder 读取
            import shutil
            from collections import defaultdict
            # 仅在未重建过时执行
            rebuilt_flag = os.path.join(val_dir, ".rebuilt_by_script")
            if not os.path.exists(rebuilt_flag):
                os.makedirs(os.path.join(val_dir, "images_reorg"), exist_ok=True)
                cls_to_files = defaultdict(list)
                with open(val_annot, "r") as f:
                    for line in f:
                        fname, cls, _, _, _, _ = line.strip().split("\t")
                        cls_to_files[cls].append(fname)
                for cls, files in cls_to_files.items():
                    cls_dir = os.path.join(val_dir, "images_reorg", cls)
                    os.makedirs(cls_dir, exist_ok=True)
                    for fname in files:
                        src = os.path.join(val_dir, "images", fname)
                        dst = os.path.join(cls_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy(src, dst)
                open(rebuilt_flag, "w").close()
            testset = torchvision.datasets.ImageFolder(
                os.path.join(val_dir, "images_reorg"), transform=transform
            )
        else:
            # 兜底：直接用 ImageFolder 读取 val 目录（若已手工整理过）
            testset = torchvision.datasets.ImageFolder(
                os.path.join(val_dir, "images"), transform=transform
            )

    elif args.dataset == 'agnews':
        max_len = 200
        trainset, testset = torchtext.datasets.AG_NEWS(root=dir_path+"rawdata")
        trainlabel, traintext = list(zip(*trainset))
        testlabel, testtext = list(zip(*testset))
        dataset_text = []
        dataset_label = []

        dataset_text.extend(traintext)
        dataset_text.extend(testtext)
        dataset_label.extend(trainlabel)
        dataset_label.extend(testlabel)

        tokenizer = get_tokenizer('basic_english')

        vocab = build_vocab_from_iterator(map(tokenizer, iter(dataset_text)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        text_pipeline = lambda x: vocab(tokenizer(x))
        # train_text_pipeline = get_vocab(traintext)
        # test_text_pipeline = get_vocab(testtext)
        label_pipeline = lambda x: int(x) - 1
        print(len(vocab))
        def text_transform(text, label,max_len=0):
            label_list, text_list = [], []
            for _text, _label in zip(text, label):
                label_list.append(label_pipeline(_label))
                text_ = text_pipeline(_text)
                padding = [0 for i in range(max_len-len(text_))]
                text_.extend(padding)
                text_list.append(text_[:max_len])
            return label_list, text_list

        train_label_list, train_text_list = text_transform(traintext, trainlabel, max_len)
        test_label_list, test_text_list = text_transform(testtext, testlabel, max_len)

        # train_text_lens = [len(text) for text in train_text_list]
        # train_text_list = [(text, l) for text, l in zip(train_text_list, train_text_lens)]
        # train_text_list = np.array(train_text_list, dtype=object)
        # train_label_list = np.array(train_label_list)

        # test_text_lens = [len(text) for text in test_text_list]
        # test_text_list = [(text, l) for text, l in zip(test_text_list, test_text_lens)]
        # test_text_list = np.array(test_text_list, dtype=object)
        # test_label_list = np.array(test_label_list)

        X_train_tensor = torch.tensor(train_text_list)
        y_train_tensor = torch.tensor(train_label_list)
        X_test_tensor = torch.tensor(test_text_list)
        y_test_tensor = torch.tensor(test_label_list)


        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)

    elif args.dataset == 'sogounews':  # 或 'sogou'
        max_len = 200
        # 关键修改：AG_NEWS -> SogouNews
        trainset, testset = torchtext.datasets.SogouNews(root=dir_path + "rawdata")
        trainlabel, traintext = list(zip(*trainset))
        testlabel, testtext = list(zip(*testset))
        dataset_text = []
        dataset_label = []

        dataset_text.extend(traintext)
        dataset_text.extend(testtext)
        dataset_label.extend(trainlabel)
        dataset_label.extend(testlabel)

        tokenizer = get_tokenizer('basic_english')

        vocab = build_vocab_from_iterator(map(tokenizer, iter(dataset_text)), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x) - 1
        print(len(vocab))

        def text_transform(text, label, max_len=0):
            label_list, text_list = [], []
            for _text, _label in zip(text, label):
                label_list.append(label_pipeline(_label))
                text_ = text_pipeline(_text)
                padding = [0 for i in range(max_len - len(text_))]
                text_.extend(padding)
                text_list.append(text_[:max_len])
            return label_list, text_list

        train_label_list, train_text_list = text_transform(traintext, trainlabel, max_len)
        test_label_list, test_text_list = text_transform(testtext, testlabel, max_len)

        X_train_tensor = torch.tensor(train_text_list)
        y_train_tensor = torch.tensor(train_label_list)
        X_test_tensor = torch.tensor(test_text_list)
        y_test_tensor = torch.tensor(test_label_list)

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)
    else:
        raise

    testdata = []

    for i in tqdm(range(len(testset)), desc="Loading test data", unit="sample"):
        testdata.append(testset[i])

    list_label2indices = classify_label(trainset, num_classes)
    try:
        all_targets = trainset.targets
    except:
        all_targets = y_train_tensor

    # Perform data partitioning based on the method determined earlier
    if data_partition == 'LDA':
        total_class_num, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices),num_classes,imb)
        list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), num_classes,num_clients,niid, seed)
    elif data_partition == 'shard':
        net_dataidx_map, rand_set_all = sharding_partition(all_targets, num_clients, args.shard_per_user)
        list_client2indices =[]
        for id in net_dataidx_map:
            list_client2indices.append(net_dataidx_map[id])
    else:
        print("check LDA and shard input!!!!")
        exit()
    original_dict_per_client, total_class_num = show_clients_data_distribution(all_targets, list_client2indices, num_classes)
    save_file(config_path, train_path, test_path, list_client2indices, trainset,testdata, num_clients, num_classes,niid,imb,seed,original_dict_per_client,total_class_num,args.shard_per_user,data_partition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-data', "--dataset", type=str, default="sogounews")
    parser.add_argument('-nc', "--num_clients", type=int, default=100)
    parser.add_argument('-if','--imb_factor', default=0.5, type=float, help='imbalance factor')

    parser.add_argument('--LDA', action='store_true', default=True)
    parser.add_argument('-noniid','--non_iid_alpha', type=float, default=0.1)

    parser.add_argument('--shard', action='store_true', default=False)
    parser.add_argument('-shard_per_user', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms

    if args.dataset == 'cinic10':
        dir_path = "data/cinic10/"
        num_classes = 10
    elif args.dataset == 'cifar10':
        dir_path = "data/cifar10/"
        num_classes = 10
    elif args.dataset == 'cifar100':
        dir_path = "data/cifar100/"
        num_classes = 100
    elif args.dataset == 'fminist':
        dir_path = "data/fminist/"
        num_classes = 10
    elif args.dataset == 'agnews':
        dir_path = "data/agnews/"
        num_classes = 4
        import torchtext
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    elif args.dataset == 'sogounews':
        dir_path = "data/sogounews/"
        num_classes = 5
        import torchtext
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    else:
        print("check your dataset")
        exit()


    generate_data(dir_path, args.num_clients, num_classes, args.non_iid_alpha,args.imb_factor,args.seed,args)