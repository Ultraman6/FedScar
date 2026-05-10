import torch
import os
import numpy as np
import csv
import copy, time
import inspect, random
import shutil, re, openpyxl  # 导入openpyxl库
from datetime import datetime
from utils.data_utils import read_data, read_client_json, read_client_data, set_data_partition_dir
from torch.utils.data import DataLoader
from utils.mem_utils import get_gpu_memory_usage
from tqdm import tqdm
from utils.flat_metrics import low_pass, Hessian, grad_norm
from utils.landscape import VisualizationConfig, LossLandscapeVisualizer


class Server(object):
    def __init__(self, args, hyperparams=None):
        """
        Args:
            args: 基础参数对象（包含device, dataset, num_classes等）
            hyperparams: 算法超参数字典，例如 {'rho': 0.1, 'beta': 0.05}
        """
        # Set up the main attributes
        self.device = args.device
        self.data_partition_dir = args.data_partition_dir
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.lr_decay = args.lr_decay
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.global_model = args.model
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.hyperparams = hyperparams
        # 设置超参数（如果提供了hyperparams，从中获取；否则设为默认值0.0）
        if hyperparams is None:
            hyperparams = {}
        for k, v in hyperparams.items():
            setattr(self, k, v)
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.test_acc = []
        self.test_loss = []
        self.personal_acc = []
        self.time = []
        self.time_whole = []
        self.memory = []
        self.speed = []
        self.current_acc = 0

        self.save_gap = args.save_gap
        self.save_model = args.save_model
        self.save_local = args.save_local
        self.eval_round = args.eval_round
        # 设置数据划分目录（如果指定了）
        if hasattr(args, 'data_partition_dir') and args.data_partition_dir:
            set_data_partition_dir(args.data_partition_dir)

        self.data_test = read_data(self.dataset, idx=None, is_train=False)
        self.test_loader = DataLoader(self.data_test, 1000, pin_memory=True, num_workers=4)
        self.loss = torch.nn.CrossEntropyLoss()

        self.loss_diff = []
        self.variance = []
        self.flat_disc = []

        # 生成结果目录名称（算法名称+超参数+时间）
        self.result_dir_name = self._generate_result_dir_name()

        self.specific_results = {}

        # 保存当前 server 类引用
        self.server_class = self.__class__
        self.client_class = None
        self.best_ckpt = None
        self.best_acc = 0.0

    def _generate_result_dir_name(self, timestamp=''):
        """
        生成结果目录名称：算法名称_超参数_时间戳
        例如: FedSAM_rho0.1_2401151430
        """
        from datetime import datetime

        # 获取当前时间戳（年份后两位，精确到分钟，无分隔符）
        if not timestamp:
            timestamp = datetime.now().strftime("%y%m%d%H%M")
        # 根据算法生成超参数字符串
        params_str = '-'.join([f'{k}={v}' for k, v in self.hyperparams.items()])
        # 组合目录名
        if params_str:
            dir_name = f"{self.algorithm}--{params_str}--{timestamp}"
        else:
            dir_name = f"{self.algorithm}--{timestamp}"

        fl_params = (f'jr={self.join_ratio}-rs={self.global_rounds}-es={self.local_epochs}'
                     f'-bs={self.batch_size}-lr={self.learning_rate}-lrd={self.lr_decay}-m={self.momentum}-wd={self.weight_decay}')
        return os.path.join(fl_params, dir_name)

    def set_clients(self, args, clientObj):
        # 保存 client 类引用
        self.client_class = clientObj
        for i in range(self.num_clients):
            train_samples = read_client_json(self.dataset, i)[0]
            client = clientObj(args, id=i,
                               train_samples=train_samples)
            for k, v in self.hyperparams.items():
                setattr(client, k, v)
            self.clients.append(client)

    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))

        return selected_clients

    def send_models(self, selected_clients, model):
        assert (len(self.clients) > 0)

        for client in selected_clients:
            client.model = copy.deepcopy(model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        receive_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []  # num of samples
        self.uploaded_models = []
        tot_samples = 0
        for client in receive_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        fedavg_global_params = self.global_model.state_dict()
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params, local_weight in zip(self.uploaded_models, self.uploaded_weights):
                list_values_param.append(dict_local_params[name_param] * local_weight)
            value_global_param = sum(list_values_param)
            fedavg_global_params[name_param] = value_global_param
        self.global_model.load_state_dict(fedavg_global_params)

    def _save_source_files(self):
        """
        保存当前使用的 server 和 client 类的源代码文件到结果目录
        """
        try:
            # 确保结果目录存在
            result_dir = os.path.join(self.data_partition_dir, self.result_dir_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # 保存 server 类源代码
            if self.server_class is not None:
                server_file = inspect.getfile(self.server_class)
                if os.path.exists(server_file):
                    server_dest = os.path.join(result_dir, os.path.basename(server_file))
                    # 如果文件不存在才保存，避免重复
                    if not os.path.exists(server_dest):
                        shutil.copy2(server_file, server_dest)
                        print(f"Server source file saved to: {server_dest}")

            # 保存 client 类源代码（如果已设置）
            if self.client_class is not None:
                client_file = inspect.getfile(self.client_class)
                if os.path.exists(client_file):
                    client_dest = os.path.join(result_dir, os.path.basename(client_file))
                    # 如果文件不存在才保存，避免重复
                    if not os.path.exists(client_dest):
                        shutil.copy2(client_file, client_dest)
                        print(f"Client source file saved to: {client_dest}")
        except Exception as e:
            print(f"Warning: Failed to save source files: {e}")

    def save_specific_results(self):
        result_dir = os.path.join(self.data_partition_dir, self.result_dir_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if len(self.specific_results):
            specific_file = os.path.join(result_dir, "specific_results.csv")
            print(f"Specific results file: {specific_file}")
            with open(specific_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # 直接保存value
                for key, value in self.specific_results.items():
                    writer.writerow([key, value])

    def save_results(self, r, save_model=True):
        """
        保存结果和模型到指定目录
        目录结构: ../results/{dataset}/{algorithm}_{hyperparams}_{timestamp}/
        """
        # 生成结果目录路径：数据集/算法_超参数_时间戳
        result_dir = os.path.join(self.data_partition_dir, self.result_dir_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_path_abs = os.path.abspath(result_dir)
        print(f"Saving results to: {result_path_abs}")

        # 保存准确率结果
        if len(self.test_acc):
            file_path = os.path.join(result_dir, 'test_accuracy.csv')
            print(f"Accuracy file: {file_path}")
            my_list_2d = [[x] for x in self.test_acc]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存准确率结果
        if len(self.test_loss):
            file_path = os.path.join(result_dir, 'test_loss.csv')
            print(f"Loss file: {file_path}")
            my_list_2d = [[x] for x in self.test_loss]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存时间结果
        if len(self.time):
            file_path = os.path.join(result_dir, 'time.csv')
            print(f"Time file: {file_path}")
            my_list_2d = [[x] for x in self.time]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存时间结果
        if len(self.time_whole):
            file_path = os.path.join(result_dir, 'time_whole.csv')
            print(f"Time file: {file_path}")
            my_list_2d = [[x] for x in self.time_whole]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存损失差异结果
        if len(self.loss_diff):
            loss_diff_file = os.path.join(result_dir, "loss_diff.csv")
            print(f"Loss diff file: {loss_diff_file}")
            my_list_2d = [[x] for x in self.loss_diff]
            with open(loss_diff_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存损失差异结果
        if len(self.flat_disc):
            flat_disc_file = os.path.join(result_dir, "flat_disc.csv")
            print(f"Flat Disc file: {flat_disc_file}")
            my_list_2d = [[x] for x in self.flat_disc]
            with open(flat_disc_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存损失差异结果
        if len(self.variance):
            variance_file = os.path.join(result_dir, "variance.csv")
            print(f"variance file: {variance_file}")
            my_list_2d = [[x] for x in self.variance]
            with open(variance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        # 保存模型
        if save_model and self.save_model:
            model_dir = os.path.join(result_dir, "models", f"round_{r}")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "global.pt")
            torch.save(self.global_model.state_dict(), model_path)
            if self.save_local: [torch.save(c.model.state_dict(),
                                    os.path.join(model_dir, f'local_{c.id}_s.pt'
                                    if c in self.selected_clients else f'local_{c.id}.pt'))
                                 for c in self.clients]
            print(f"Model saved to: {model_dir}")
            # self.save_best_model()

    def save_best_model(self):
        """
        保存最佳模型
        模型保存到: ../results/{dataset}/{algorithm}_{hyperparams}_{timestamp}/models/round_best.pt
        """
        # 使用与save_results相同的目录结构
        result_dir = os.path.join(self.data_partition_dir, self.result_dir_name)
        model_dir = os.path.join(result_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        best_model_path = os.path.join(model_dir, "round_best.pt")
        torch.save(self.best_ckpt, best_model_path)
        print(f"Best model with test acc: {self.best_acc} saved to: {best_model_path}")

    # evaluate
    def evaluate(self):
        model = self.global_model
        model.eval()
        with torch.no_grad():
            num_corrects = {i: 0 for i in range(self.num_classes)}
            total = {i: 0 for i in range(self.num_classes)}
            total_corrects = 0
            total_samples = 0
            total_loss = 0
            for step, data_batch in enumerate(self.test_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicts = torch.max(outputs, -1)
                for i in range(len(labels)):
                    total[labels[i].item()] += 1
                    total_samples += 1
                    if predicts[i] == labels[i]:
                        num_corrects[labels[i].item()] += 1
                        total_corrects += 1

            test_loss = round(total_loss / len(self.test_loader.dataset), 5)
            print(f"test loss: {test_loss}, lr: {self.learning_rate}")

            total_accuracy = total_corrects / total_samples
            self.test_acc.append(total_accuracy)
            self.test_loss.append(test_loss)
            # all_model_tensor = torch.cat([param_to_vector(c.model) for c in self.clients])
            # self.model_var.append(torch.var(all_model_tensor).item())
            print(f"Global acc:{self.test_acc}")
            # print(f"Model var:{self.model_var}")
            if total_accuracy >= self.best_acc:
                self.best_acc = total_accuracy
                self.best_ckpt = copy.deepcopy(self.global_model.state_dict())
        return test_loss

    # evaluate
    def validation(self, dataloader):
        model = self.global_model
        model.eval()
        with torch.no_grad():
            num_corrects = {i: 0 for i in range(self.num_classes)}
            total = {i: 0 for i in range(self.num_classes)}
            total_corrects = 0
            total_samples = 0
            total_loss = 0
            for step, data_batch in enumerate(dataloader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicts = torch.max(outputs, -1)
                for i in range(len(labels)):
                    total[labels[i].item()] += 1
                    total_samples += 1
                    if predicts[i] == labels[i]:
                        num_corrects[labels[i].item()] += 1
                        total_corrects += 1
            loss = round(total_loss / len(self.test_loader.dataset), 5)
            accuracy = total_corrects / total_samples
        model.train()
        return loss, accuracy

    def _lr_scheduler_(self):
        self.learning_rate *= self.lr_decay

    def sharpness(self):
        model = self.global_model
        model.eval()
        with torch.no_grad():
            python_rng_state = random.getstate()  # 绝大部分随机Transform的核心依赖
            torch_cpu_rng_state = torch.get_rng_state()  # PyTorch专属Transform依赖
            torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            numpy_rng_state = np.random.get_state()  # 自定义numpy随机Transform依赖
            try:
                train_data = []
                for client in self.selected_clients:
                    train_data += read_client_data(self.dataset, client.id)
                train_loader = DataLoader(train_data, 500, shuffle=True, pin_memory=True)
                total_loss = 0
                correct = 0.0
                for step, data_batch in enumerate(train_loader):
                    images, labels = data_batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = self.loss(outputs, labels)
                    total_loss += loss.item() * labels.size(0)
                    pred = outputs.data.argmax(1, keepdim=True)
                    correct += pred.eq(labels.data.view_as(pred)).sum().item()
            finally:
                random.setstate(python_rng_state)
                torch.set_rng_state(torch_cpu_rng_state)
                if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(torch_cuda_rng_state)
                np.random.set_state(numpy_rng_state)
        return total_loss / len(train_loader.dataset)

    def flatness_discrepancy(self):
        model = self.global_model
        model.eval()
        local_losses, total_loss, total_samples = [], 0.0, 0
        with torch.no_grad():
            for client in self.selected_clients:
                local_loss = 0.0
                local_model = client.model
                local_model.eval()
                python_rng_state = random.getstate()  # 绝大部分随机Transform的核心依赖
                torch_cpu_rng_state = torch.get_rng_state()  # PyTorch专属Transform依赖
                torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                numpy_rng_state = np.random.get_state()  # 自定义numpy随机Transform依赖
                try:
                    train_loader = DataLoader(read_client_data(self.dataset, client.id),
                                              500, shuffle=True, pin_memory=True)
                    for step, data_batch in enumerate(train_loader):
                        images, labels = data_batch
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        local_outputs = local_model(images)
                        loss = self.loss(outputs, labels)
                        loss_ = self.loss(local_outputs, labels)
                        total_loss += loss.item() * labels.size(0)
                        local_loss += loss_.item() * labels.size(0)
                    total_samples += len(train_loader.dataset)
                    local_losses.append(local_loss)
                finally:
                    random.setstate(python_rng_state)
                    torch.set_rng_state(torch_cpu_rng_state)
                    if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(torch_cuda_rng_state)
                    np.random.set_state(numpy_rng_state)
        self.flat_disc.append(abs(total_loss - sum(local_losses)) / total_samples)
        return total_loss / total_samples

    def sharpness_v2(self, time_str, round_list):
        self.result_dir_name = self._generate_result_dir_name(time_str)
        model_dir = os.path.join(self.data_partition_dir, self.result_dir_name, "models")
        self.global_model.train()

        # 1. 初始化Excel
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sharpness_Results"
        header = ["round", "val loss", "val acc", "val norm",
                  "test loss", "test acc", "test norm",
                  "lpf", "trace"]
        ws.append(header)

        # -------------------------------------------------------------
        # 核心修改：构建待处理的任务列表 [(round_val, filename), ...]
        # -------------------------------------------------------------
        tasks = []

        # 情况 A: 指定了 round_list -> 直接拼接查找，不使用正则
        if round_list and len(round_list) > 0:
            for r in round_list:
                filename = f"round_{r}.pt"
                file_path = os.path.join(model_dir, filename)

                if os.path.exists(file_path):
                    # r 可能是 int 也可能是 'best'，直接保留原值
                    tasks.append((r, filename))
                else:
                    print(f"Warning: 指定的模型文件 {filename} 不存在，跳过。")

        # 情况 B: round_list 为空 -> 遍历目录下所有符合格式的文件
        else:
            # 这里仍需正则来识别哪些文件是模型，并提取 round 信息
            pattern = re.compile(r"round_(best|\d+)\.pt")
            # 排序文件名，保证 Excel 写入顺序整洁
            all_files = sorted(os.listdir(model_dir))

            for filename in all_files:
                match = pattern.match(filename)
                if match:
                    val_str = match.group(1)
                    # 尝试将数字转为 int，'best' 保持为 string
                    try:
                        r = int(val_str)
                    except ValueError:
                        r = val_str  # 处理 'best' 的情况

                    tasks.append((r, filename))

        # -------------------------------------------------------------
        # 统一处理任务列表
        # -------------------------------------------------------------
        for r, filename in tasks:
            print(f"Processing: {filename}")
            model_path = os.path.join(model_dir, filename)

            # 加载模型
            state_dict = torch.load(model_path)
            self.global_model.load_state_dict(state_dict)

            # 计算锐度
            s1, s2, s3, s4, s5, s6, s7, s8 = self.cal_sharpness()

            # 组装数据
            row_data = [r, s1, s2, s3, s4, s5, s6, s7, s8]
            ws.append(row_data)

        # 4. 保存Excel文件
        excel_filename = "sharpness_results.xlsx"
        excel_path = os.path.join(model_dir, excel_filename)
        wb.save(excel_path)
        wb.close()
        print(f"锐度结果已保存至：{excel_path}")

    def cal_sharpness(self):
        train_data = []
        for client in self.clients:
            train_data += read_client_data(self.dataset, client.id)
        train_loader = DataLoader(train_data, 500, shuffle=True, pin_memory=True)
        hessian_comp = Hessian(self.global_model, self.loss, self.device, dataloader=train_loader)
        python_rng_state = random.getstate()  # 绝大部分随机Transform的核心依赖
        torch_cpu_rng_state = torch.get_rng_state()  # PyTorch专属Transform依赖
        torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        numpy_rng_state = np.random.get_state()  # 自定义numpy随机Transform依赖
        try:
            val_loss, val_acc = self.validation(train_loader)
            test_loss, test_acc = self.validation(self.test_loader)
        finally:
            random.setstate(python_rng_state)
            torch.set_rng_state(torch_cpu_rng_state)
            if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            np.random.set_state(numpy_rng_state)
        try:
            val_norm = grad_norm(self.device, self.global_model, train_loader, self.loss)
            test_norm = grad_norm(self.device, self.global_model, self.test_loader, self.loss)
        finally:
            random.setstate(python_rng_state)
            torch.set_rng_state(torch_cpu_rng_state)
            if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            np.random.set_state(numpy_rng_state)
        try:
            lpf = low_pass(self.global_model, train_loader, self.loss, 0.01, 100)
        finally:
            random.setstate(python_rng_state)
            torch.set_rng_state(torch_cpu_rng_state)
            if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            np.random.set_state(numpy_rng_state)
        try:
            trace = np.mean(hessian_comp.trace())
        finally:
            random.setstate(python_rng_state)
            torch.set_rng_state(torch_cpu_rng_state)
            if torch.cuda.is_available() and torch_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            np.random.set_state(numpy_rng_state)
        return val_loss, val_acc, val_norm, test_loss, test_acc, test_norm, lpf, trace

    def _generate_loss_landscape_subdir_name(self, x_range, y_range, direction_type, direction_method,
                                             x_norm, y_norm, ignore_bias_bn, use_amp,
                                             top_n=None, project=None, lstsq_lr=None, lstsq_epochs=None,
                                             trajectory_display='none', auto_coords=False):
        """
        生成损失景观子目录名称（与 LossLandscapeVisualizer.generate_filename 格式一致）

        Args:
            x_range: X轴范围 (start, end, points)
            y_range: Y轴范围 (start, end, points) 或 None
            direction_type: 方向类型 ('weights' 或 'states')
            direction_method: 方向方法 ('random', 'interpolate', 'pca', 'eigen', 'lstsq')
                - 'pca': 纯PCA方向
                - 'lstsq': PCA + Fitting Error Optimization（论文方法）
            x_norm: X方向归一化方式
            y_norm: Y方向归一化方式
            ignore_bias_bn: 是否忽略 bias 和 BN 参数
            use_amp: 是否使用混合精度
            top_n: top_n 参数（用于 eigen/pca）
            project: project 参数（用于 pca）
            lstsq_lr: lstsq 学习率（用于 lstsq）
            lstsq_epochs: lstsq 训练轮数（用于 lstsq）
            trajectory_display: 轨迹可视化模式 ('none', 'points', 'trajectory')
            auto_coords: 是否启用动态坐标

        Returns:
            str: 子目录名称
        """
        # 获取坐标信息
        x_start, x_end, x_points = x_range
        dim_str = "1d" if y_range is None else "2d"

        # 构建文件名组件
        components = [
            dim_str,  # 直接使用 "2d" 或 "1d"，不加前缀
            direction_method,  # 'pca', 'lstsq', 'random', 'eigen' 等
            f"xnorm_{x_norm}",  # 保留 "xnorm_" 前缀
            f"ynorm_{y_norm}",  # 保留 "ynorm_" 前缀
        ]

        # 根据 auto_coords 决定坐标范围的表示方式
        if auto_coords:
            # 动态坐标模式：只显示点数
            components.append(f"auto_coords_{x_points}pts")
        else:
            # 固定坐标模式：显示完整范围
            components.append(f"x{x_start:.1f}to{x_end:.1f}_{x_points}pts")
            # 添加 y 范围信息（2D 可视化）
            if y_range is not None:
                y_start, y_end, y_points = y_range
                components.append(f"y{y_start:.1f}to{y_end:.1f}_{y_points}pts")

        # 添加方向特定的参数
        if direction_method == 'eigen' and top_n is not None:
            components.append(f"top{top_n}")
        elif direction_method == 'pca':
            if top_n is not None:
                components.append(f"top{top_n}")
            if project is not None:
                components.append(f"proj_{project}")
        elif direction_method == 'lstsq':
            if lstsq_lr is not None:
                components.append(f"lr{lstsq_lr}")
            if lstsq_epochs is not None:
                components.append(f"epochs{lstsq_epochs}")

        # 添加轨迹可视化模式
        components.append(f"traj_{trajectory_display}")

        # 添加其他重要参数
        if ignore_bias_bn:
            components.append("nobiasbn")

        if use_amp:
            components.append("amp")

        # 添加时间戳以避免冲突
        timestamp = int(time.time())
        components.append(f"t{timestamp}")

        # 连接组件
        subdir_name = "_".join(components)

        return subdir_name

    def visualize_loss_landscape(self, time_str, round_list):
        """
        使用 loss landscape 可视化工具可视化损失景观
        所有参数使用默认值，从 server 对象获取模型、损失函数、设备和训练数据。

        Args:
            time_str: 时间戳字符串，用于生成结果目录名称
            round_list: 要处理的轮次列表；在 PCA 模式下仅用于决定可视化哪个轮次，
                        PCA 方向本身会基于 model_dir 下所有 round_xx.pt 模型轨迹自动估计

        Returns:
            dict: 包含可视化结果的字典
        """
        # 使用 time_str 生成结果目录名称（参考 sharpness_v2）
        self.result_dir_name = self._generate_result_dir_name(time_str)

        # 从 server 对象获取模型、损失函数、设备
        model = self.global_model
        criterion = self.loss
        device = self.device

        # 收集所有客户端的训练数据
        train_data = []
        for client in self.clients:
            train_data += read_client_data(self.dataset, client.id)

        # 设置默认参数值（与 argparse 默认值保持一致）
        batch_size = 1000
        num_workers = 4

        # 创建训练数据加载器
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers
        )

        # 设置损失景观相关参数的默认值（与 argparse 默认值保持一致）
        x_range = (-30.0, 30.0, 51)  # 对应 argparse: default=[-1.0, 1.0, 51]
        y_range = (-30.0, 30.0, 51)  # 对应 argparse: default=[-1.0, 1.0, 51] (2D visualization)
        direction_type = 'weights'
        # direction_method: 'pca' = 纯PCA, 'lstsq' = PCA + Fitting Error Optimization（论文方法）
        direction_method = 'lstsq'  # 使用 PCA + Fitting 模式
        x_norm = 'filter'
        y_norm = 'filter'
        ignore_bias_bn = True
        end_root = None
        trajectory = None
        project = None
        top_n = 2
        lstsq_lr = 0.004
        lstsq_epochs = 10
        use_amp = True
        save_format = 'h5'
        plot_style = 'default'
        dpi = 300
        show_plots = False

        # 轨迹可视化参数
        trajectory_display = 'points'  # 'none', 'points', 'trajectory'
        loss_max_threshold = None  # 损失阈值截断（可选）

        # 动态坐标计算参数
        auto_coords = True  # 启用动态坐标（根据轨迹投影自动确定x/y范围）
        coord_expand_ratio = 0.5  # 边界扩展比例（0.5表示向外扩展50%）

        # trajectory 来源参数
        # 'local': 从 models/round_xx/ 目录下的 local_xx_s.pt 文件构建 trajectory
        # 'global': 从 models/ 目录下的 round_xx.pt 文件构建 trajectory
        trajectory_source = 'local'

        # 设置保存目录：参考其他方法的 result_dir 构造方式
        result_dir = os.path.join(self.data_partition_dir, self.result_dir_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 模型目录（与 save_results 中保持一致）
        model_dir = os.path.join(result_dir, "models")

        # -------------------------------------------------------------
        # 构建 PCA 需要的模型轨迹（trajectory）
        # 注意：trajectory 将在每个 round 处理时动态构建
        # -------------------------------------------------------------

        # 生成子目录名称（基于参数值组合）
        subdir_name = self._generate_loss_landscape_subdir_name(
            x_range=x_range,
            y_range=y_range,
            direction_type=direction_type,
            direction_method=direction_method,
            x_norm=x_norm,
            y_norm=y_norm,
            ignore_bias_bn=ignore_bias_bn,
            use_amp=use_amp,
            top_n=top_n,
            project=project,
            lstsq_lr=lstsq_lr,
            lstsq_epochs=lstsq_epochs,
            trajectory_display=trajectory_display,
            auto_coords=auto_coords
        )

        # 如果指定了 round_list，处理每个 round 的模型
        if round_list and len(round_list) > 0:
            results_list = []

            for r in round_list:
                # 根据 trajectory_source 决定模型加载路径
                if trajectory_source == 'local':
                    # local 模式：从 models/round_{r}/global.pt 加载
                    round_dir = os.path.join(model_dir, f"round_{r}")
                    current_model_path = os.path.join(round_dir, "global.pt")
                    model_desc = f"round_{r}/global.pt"
                else:  # trajectory_source == 'global'
                    # global 模式：从 models/round_{r}.pt 加载
                    round_dir = os.path.join(model_dir, f"round_{r}")  # 仍用于 local trajectory（如果需要）
                    current_model_path = os.path.join(model_dir, f"round_{r}.pt")
                    model_desc = f"round_{r}.pt"

                if not os.path.exists(current_model_path):
                    print(f"Warning: 指定的模型文件 {model_desc} 不存在，跳过。")
                    continue

                print(f"Processing loss landscape for: {model_desc}")

                # 加载当前模型
                state_dict = torch.load(current_model_path, map_location=device)
                model.load_state_dict(state_dict)

                # 为当前 round 构建 trajectory（PCA 和 lstsq 模式都需要）
                round_trajectory = None
                if direction_method in ['pca', 'lstsq']:
                    if trajectory_source == 'local':
                        # 从 round_{r} 目录加载 trajectory：local_xx_s.pt 文件
                        if os.path.exists(round_dir):
                            round_trajectory = self._load_trajectory_from_round_dir(round_dir)
                            if not round_trajectory:
                                print(f"Warning: 无法从 {round_dir} 构建有效的模型轨迹，方向估计可能失败。")
                        else:
                            print(f"Warning: Round 目录 {round_dir} 不存在，无法构建模型轨迹。")
                    else:  # trajectory_source == 'global'
                        # 从 models/ 目录加载 trajectory：round_xx.pt 文件
                        round_trajectory = self._load_trajectory_from_global_models(model_dir)
                        if not round_trajectory:
                            print(f"Warning: 无法从 {model_dir} 构建有效的模型轨迹，方向估计可能失败。")
                else:
                    round_trajectory = trajectory  # 其他模式使用全局 trajectory

                # 在 result_dir 下创建 loss_landscape 子目录，以 round 命名，然后添加参数子目录
                save_dir = os.path.join(result_dir, 'loss_landscape', f'round_{r}', subdir_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 创建可视化配置
                config = VisualizationConfig(
                    x_range=x_range,
                    y_range=y_range,
                    auto_coords=auto_coords,
                    coord_expand_ratio=coord_expand_ratio,
                    direction_type=direction_type,
                    direction_method=direction_method,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    ignore_bias_bn=ignore_bias_bn,
                    end_root=end_root,
                    trajectory=round_trajectory,
                    project=project,
                    top_n=top_n,
                    lstsq_lr=lstsq_lr,
                    lstsq_epochs=lstsq_epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_amp=use_amp,
                    save_dir=save_dir,
                    save_format=save_format,
                    plot_style=plot_style,
                    dpi=dpi,
                    show_plots=show_plots,
                    loss_max_threshold=loss_max_threshold,
                    trajectory_display=trajectory_display
                )

                # 创建可视化器
                visualizer = LossLandscapeVisualizer(
                    model, train_loader, criterion, device, config
                )

                # 执行可视化
                print(f"Starting loss landscape visualization for round {r}...")
                results = visualizer.visualize(
                    save_results=True,
                    save_plots=True,
                    filename='xyz'
                )

                print(f"Loss landscape visualization for round {r} completed!")
                print(f"Results saved to: {results['results_file']}")
                results_list.append((r, results))

            return results_list
        else:
            # 使用当前模型进行可视化
            # 在 result_dir 下创建 loss_landscape 子目录，然后添加参数子目录
            save_dir = os.path.join(result_dir, 'loss_landscape', subdir_name)

            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 创建可视化配置
            config = VisualizationConfig(
                x_range=x_range,
                y_range=y_range,
                auto_coords=auto_coords,
                coord_expand_ratio=coord_expand_ratio,
                direction_type=direction_type,
                direction_method=direction_method,
                x_norm=x_norm,
                y_norm=y_norm,
                ignore_bias_bn=ignore_bias_bn,
                end_root=end_root,
                trajectory=trajectory,
                project=project,
                top_n=top_n,
                lstsq_lr=lstsq_lr,
                lstsq_epochs=lstsq_epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                use_amp=use_amp,
                save_dir=save_dir,
                save_format=save_format,
                plot_style=plot_style,
                dpi=dpi,
                show_plots=show_plots,
                loss_max_threshold=loss_max_threshold,
                trajectory_display=trajectory_display
            )

            # 创建可视化器
            visualizer = LossLandscapeVisualizer(
                model, train_loader, criterion, device, config
            )

            # 执行可视化
            print("Starting loss landscape visualization...")
            results = visualizer.visualize(
                save_results=True,
                save_plots=True,
                filename='xyz'
            )

            print(f"Loss landscape visualization completed!")
            print(f"Results saved to: {results['results_file']}")

            return results

    def _load_trajectory_from_round_dir(self, round_dir):
        """
        从 round 目录加载模型轨迹（用于PCA/lstsq模式）
        只加载 local_xx_s.pt 模型（SAM扰动后的模型）作为trajectory

        Args:
            round_dir: round 目录路径，例如 models/round_0

        Returns:
            List[nn.Module]: 模型轨迹列表，只包含 local_xx_s.pt 文件
        """
        import re
        trajectory = []

        if not os.path.isdir(round_dir):
            print(f"Warning: {round_dir} 不是有效目录")
            return trajectory

        # 只加载 local_xx_s.pt 文件（SAM扰动后的模型）
        # 使用正则表达式匹配：local_数字_s.pt
        local_s_pattern = re.compile(r'^local_\d+_s\.pt$')

        all_files = os.listdir(round_dir)
        local_s_files = [f for f in all_files if local_s_pattern.match(f)]
        local_s_files.sort()  # 排序以保证顺序一致

        for local_file in local_s_files:
            local_file_path = os.path.join(round_dir, local_file)
            try:
                checkpoint = torch.load(local_file_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                temp_model = copy.deepcopy(self.global_model)
                temp_model.load_state_dict(state_dict)
                trajectory.append(temp_model)
                print(f"Loaded local_s model from: {local_file}")
            except Exception as e:
                print(f"Warning: Failed to load local_s model from {local_file_path}: {e}")

        if not trajectory:
            print(f"Warning: No valid local_s models loaded from {round_dir}")
        else:
            print(f"Successfully loaded {len(trajectory)} local_s models for trajectory")

        return trajectory

    def _load_trajectory_from_path(self, trajectory_path):
        """
        从路径加载模型轨迹（用于PCA/lstsq方法，兼容旧格式）

        Args:
            trajectory_path: 模型文件路径或包含模型文件的目录路径

        Returns:
            List[nn.Module]: 模型轨迹列表
        """
        import glob
        trajectory = []

        if os.path.isfile(trajectory_path):
            # 单个文件
            try:
                checkpoint = torch.load(trajectory_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # 创建临时模型并加载状态
                temp_model = copy.deepcopy(self.global_model)
                temp_model.load_state_dict(state_dict)
                trajectory.append(temp_model)
            except Exception as e:
                print(f"Warning: Failed to load model from {trajectory_path}: {e}")
        elif os.path.isdir(trajectory_path):
            # 目录，查找所有模型文件
            model_files = []
            for ext in ['*.pth', '*.pt', '*.ckpt']:
                model_files.extend(glob.glob(os.path.join(trajectory_path, '**', ext), recursive=True))

            for model_file in sorted(model_files):
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint

                    temp_model = copy.deepcopy(self.global_model)
                    temp_model.load_state_dict(state_dict)
                    trajectory.append(temp_model)
                except Exception as e:
                    print(f"Warning: Failed to load model from {model_file}: {e}")

        if not trajectory:
            print(f"Warning: No valid models loaded from {trajectory_path}")

        return trajectory

    def _load_trajectory_from_global_models(self, model_dir):
        """
        从 models/ 目录下的 round_xx.pt 文件构建 trajectory

        Args:
            model_dir: 模型目录路径，例如 results/.../models

        Returns:
            List[nn.Module]: 模型轨迹列表，包含所有 round_xx.pt 文件
        """
        import re
        trajectory = []

        if not os.path.isdir(model_dir):
            print(f"Warning: {model_dir} 不是有效目录")
            return trajectory

        # 匹配 round_数字.pt 文件（不包括 round_best.pt）
        round_pattern = re.compile(r'^round_(\d+)\.pt$')

        all_files = os.listdir(model_dir)
        round_files = []
        for f in all_files:
            match = round_pattern.match(f)
            if match:
                round_num = int(match.group(1))
                round_files.append((round_num, f))

        # 按 round 数字排序
        round_files.sort(key=lambda x: x[0])

        for round_num, round_file in round_files:
            round_file_path = os.path.join(model_dir, round_file)
            try:
                checkpoint = torch.load(round_file_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                temp_model = copy.deepcopy(self.global_model)
                temp_model.load_state_dict(state_dict)
                trajectory.append(temp_model)
                print(f"Loaded global round model from: {round_file}")
            except Exception as e:
                print(f"Warning: Failed to load global round model from {round_file_path}: {e}")

        if not trajectory:
            print(f"Warning: No valid round_xx.pt models loaded from {model_dir}")
        else:
            print(f"Successfully loaded {len(trajectory)} global round models for trajectory")

        return trajectory

    def empty_cache(self):
        allocated, reserved = get_gpu_memory_usage(self.device)
        print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1).detach()))
    return torch.cat(vec)
