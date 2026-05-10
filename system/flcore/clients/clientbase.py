import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import copy


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples,  **kwargs):
        self.model = None
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.weight_decay = args.weight_decay
        self.local_epochs = args.local_epochs
       
        self.loss = nn.CrossEntropyLoss()
        self.momentum = args.momentum
        self.train_data = None
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-5,momentum=0.9)

        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda epoch: self.learning_rate)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if self.train_data is None:
            self.train_data = read_client_data(self.dataset, self.id)

        return DataLoader(self.train_data, batch_size, shuffle=True,pin_memory=True)

        
    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())
    
        # self.scheduler.step()
   



