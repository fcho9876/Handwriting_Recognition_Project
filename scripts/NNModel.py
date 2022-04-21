from numpy import TooHardError
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn
import torch.nn.functional as F

class NNModel():

    # TODO Add training settings
    def __init__(self):
        self.batch_size = 128
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    # TODO Download training data
    def downloadTrainingData(self):
        self.train_dataset = datasets.MNIST(root = 'MNIST_Data_Train/', 
                                            train = True, 
                                            transform = transforms.ToTensor(), 
                                            download = True)
    # TODO Download testing data
    def downloadTestData(self):
        self.test_dataset = datasets.MNIST(root = 'MNIST_Data_Test/', 
                                            train = False, 
                                            transform = transforms.ToTensor())

    # TODO Data loader
    def dataLoader(self):
        self.train_loader = data.DataLoader(dataset = self.train_dataset, 
                                            batch_size = self.batch_size, 
                                            shuffle = True)

        self.test_loader = data.DataLoader(dataset = self.train_dataset, 
                                            batch_size = self.batch_size, 
                                            shuffle = False)

    # TODO Define first model

    # TODO Define second model

    # TODO Train model 

    # TODO Test model


