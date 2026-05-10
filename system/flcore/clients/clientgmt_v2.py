import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn

class clientGMTV2(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.EMA = None
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.dual_variable = None

    def train(self):
        # ema model accumulated locally
        EMA_para = param_to_vector(self.EMA).detach()
        global_para = param_to_vector(self.model).detach()
        EMA_para = EMA_para * self.alpha + global_para * (1 - self.alpha)
        vector_to_param(EMA_para, self.EMA)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        trainloader = self.load_train_data()
        self.model.train()

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        for params in self.EMA.parameters():
            params.requires_grad = False

        regular_params = param_to_vector(self.model).detach()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                with torch.no_grad():
                    output_t = self.EMA(x)
                output = self.model(x)
                loss = self.loss(output, y)
                # glotra loss
                pred_probs = F.log_softmax(output / self.tau, dim=1)
                dg_probs = torch.softmax(output_t / self.tau, dim=1)
                loss += self.gama * self.tau ** 2 * self.KLDiv(pred_probs, dg_probs)
                # dyn
                local_params = param_to_vector(self.model)
                loss += self.beta * torch.dot(local_params, self.dual_variable)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # DYN
        local_params = param_to_vector(self.model).detach()
        self.local_update = (local_params - regular_params)


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

