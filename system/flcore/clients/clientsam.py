import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
from utils.optim_utils import ESAM
import torch

class clientSAM(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.local_vec = None
        
    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,momentum=self.momentum)
        optimizer = ESAM(self.model.parameters(), base_optimizer, rho=self.rho)
        trainloader = self.load_train_data()
        self.model.train()
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        origin_params = param_to_vector(self.model).detach()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                optimizer.paras = [x, y, self.loss, self.model]
                optimizer.step()
                base_optimizer.step()
        regular_params = param_to_vector(self.model).detach()
        self.local_vec = regular_params - origin_params

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)

def vector_to_param(vector, model):
    # vector ---> model parameters
    vector = vector.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()

        param.data = vector[index:index + param_size].view(param.shape)

        index += param_size

def grad_to_vector(model):
    # model gradients ---> vector (same storage)
    vec = []
    for param in model.parameters():
        if param.grad is None:
            vec.append(torch.zeros_like(param).reshape(-1))
        else:
            vec.append(param.grad.reshape(-1))
    return torch.cat(vec)