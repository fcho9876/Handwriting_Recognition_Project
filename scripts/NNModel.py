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
        
        # set up model
        self.model = Net()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.5)

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

    # TODO Train model
    def train_model(self):
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()


    # TODO Test model
    def test_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1, keepdmin = True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()


# TODO Define first model
# Adapted from lab
class Net(nn.Module):
    def __init__(self):
        self.l1 = nn.linear(784, 520)
        self.l2 = nn.linear(520, 320)
        self.l3 = nn.linear(320, 240)
        self.l4 = nn.linear(240, 120)
        self.l5 = nn.lienar(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


 # TODO Define second model


