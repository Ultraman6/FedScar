from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import torch
from utils.optim_utils import ERM, ESAM

class clientInit(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

    def train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay,momentum=self.momentum)
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
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                optimizer.step()
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

def vec_add_to_grad(vec, model):
    # vec ---> grad addition
    vec = vec.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()
        param.grad.data += vec[index:index + param_size].view(param.shape)
        index += param_size