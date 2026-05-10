import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

class clientDyn(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.dual_vec = None
        self.local_vec = None

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
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                output = self.model(x)
                loss = self.loss(output, y)
                #dyn
                local_params = param_to_vector(self.model)
                loss += self.alpha/2 * torch.norm(local_params - regular_params, 2)
                loss += torch.dot(local_params, self.dual_vec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # DYN
        self.local_vec = param_to_vector(self.model).detach()-regular_params
        self.dual_vec += self.alpha * self.local_vec

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)



