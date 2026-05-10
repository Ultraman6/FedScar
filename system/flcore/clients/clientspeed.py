import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.optim_utils import ESAM

class clientSPEED(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.dual_vec = None

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay,momentum=self.momentum)
        trainloader = self.load_train_data()
        self.model.train()
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        regular_params = param_to_vector(self.model).detach()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                params = param_to_vector(self.model).detach()
                grad1 = grad_to_vector(self.model).detach()
                vector_to_param(params + grad1*(self.rho/grad1.norm(p=2)), self.model)

                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                vec_add_to_grad(self.beta*(self.dual_vec + params - regular_params), self.model)
                vector_to_param(params, self.model)
                optimizer.step()

        local_params = param_to_vector(self.model).detach()
        self.dual_vec += local_params-regular_params
        vector_to_param(local_params+self.dual_vec, self.model)

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

def vec_add_to_grad(vec, model):
    # vec ---> grad addition
    vec = vec.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()
        param.grad.data += vec[index:index + param_size].view(param.shape)
        index += param_size