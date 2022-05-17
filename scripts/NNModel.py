from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn
import torch.nn.functional as F
import time

from PIL import Image, ImageOps, ImageFilter
import numpy as np

import torchvision.models as models
import torchvision.transforms.functional as FT

import torch

class NNModel():

    def __init__(self):
        super(NNModel, self).__init__()
        self.batch_size = 64
        self.check_cancel(False)

        if cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # set up model
        self.model = Default_Net()
        #self.model = CNN()
        #self.model = ResNet()
        

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.5)

    # Cancel option either true or false
    def check_cancel(self, option):
        self.cancel = option

    def save_epoch_number(self, epoch_input):
        self.epoch_number = epoch_input
    
        

    def check_epoch(self, number):
        epoch_num = number
        if epoch_num == 1:
                return 'epoch_1'
        elif epoch_num == 2:
                return 'epoch_2'
        elif epoch_num == 3:
                return 'epoch_3'
        elif epoch_num == 4:
                return 'epoch_4'
        elif epoch_num == 5:
                return 'epoch_5'
        elif epoch_num == 6:
                return 'epoch_6'
        elif epoch_num == 7:
                return 'epoch_7'
        elif epoch_num == 8:
                return 'epoch_8'


    # set which model to be trained on
    def set_model_to_train(self, type):
        if type == "Default_Net":
            self.model = Default_Net()
        elif type == "CNN_Net":
            self.model = CNN()
        elif type == "ResNet_Net":
            self.model = ResNet()

    # Download EMNIST training and testing datasets
    # If already downloaded, load the dataset
    def download_training_dataset(self):
        self.train_dataset = datasets.EMNIST(root = 'EMNIST_Data_Train_balanced/', 
        #self.train_dataset = datasets.EMNIST(root = 'EMNIST_Data_Train/',
                                            train = True,
                                            split = 'balanced', 
                                            #split = 'byclass',
                                            transform = transforms.ToTensor(), 
                                            download = True) 

    def load_training_dataset(self):
        self.train_loader = data.DataLoader(dataset = self.train_dataset, 
                                            batch_size = self.batch_size, 
                                            shuffle = True) 

    def download_testing_dataset(self):
        self.test_dataset = datasets.EMNIST(root = 'EMNIST_Data_Test_balanced/',
        #self.test_dataset = datasets.EMNIST(root = 'EMNIST_Data_Test/', 
                                            train = False, 
                                            split = 'balanced',
                                            #split = 'byclass',
                                            transform = transforms.ToTensor(),
                                            download = True) 
    
    def load_testing_dataset(self):
        self.test_loader = data.DataLoader(dataset = self.test_dataset,
                                          batch_size = self.batch_size,
                                          shuffle = False)   

    # train model for one epoch
    def train_epoch(self):
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    # Adapted from COMPSYS 302 PyTorch Lab
    def train_model(self, epoch):
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):

            # check if process is cancelled
            if self.cancel == True:
                break

            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))

    # Adapted from COMPSYS 302 PyTorch Lab
    def test_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            test_loss += self.criterion(output, target).item()
            #pred = output.data.max(1, keepdmin = True)[1]
            pred = output.data.max(1, True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} '
            f'({100. * correct / len(self.test_loader.dataset):.0f}%)')


    def load_model_1(self):
        self.model = Default_Net()
        self.model.load_state_dict(torch.load('saved_models\Default_Net'))

    def load_model_2(self):
        self.model = CNN()
        self.model.load_state_dict(torch.load('saved_models\CNN_Net'))
        self.model.eval()

    def load_model_3(self):
        self.model = ResNet()
        self.model.load_state_dict(torch.load('saved_models\ResNet_Net'))
        self.model.eval()

    ###### Following methods are used to load model based on what the user has trained their custom model on ######
    def load_model_4_Default(self):
        try:
            self.model = Default_Net()
            self.model.load_state_dict(torch.load('saved_models\Custom_Net'))
            self.model.eval()
        except:
            # if custom model is not found, return 0. This will be used as indicator in myGUI that custom model is missing
            print("Custom_Net does not exist")
            return 0     

    def load_model_4_CNN(self):
        try:
            self.model = CNN()
            self.model.load_state_dict(torch.load('saved_models\Custom_Net'))
            self.model.eval()
        except:
            print("Custom_Net does not exist")
            return 0

    def load_model_4_ResNet(self):
        try:
            self.model = ResNet()
            self.model.load_state_dict(torch.load('saved_models\Custom_Net'))
            self.model.eval()
        except:
            print("Custom_Net does not exist")
            return 0


    # process image before fed into a train model
    def process_input_image(self):
        # original image is stored in a 400 x 400 pixel 
        #img = img.resize((200, 200))  # could improve speed of prediction by reducing image size
        img = Image.open('images/loadedimage.png')

        
        # apply gaussian blur/fliter with sigma = 1
        image_blurred = img.filter(ImageFilter.GaussianBlur(radius = 1))
        image_blurred.save('images/(b)_Gaussian_Blur.png')

        # extract ROI from our initial inputted image
        # use nested for loop to go by row and column to find blank space (contains zeros only)

        image_ROI = image_blurred.convert('L')              # convert image to black and white
        image_ROI = np.array(image_ROI)                     # convert to a numpy array data
        image_ROI = np.invert(image_ROI)                    # inverts the array

        # set up dimensions of image shape and define arrays to store position of zeros
        numRows, numCols = image_ROI.shape
        zero_row_array = []
        zero_col_array = []
        for i in range(0, numRows - 1):
            for j in range(0, numCols):
                if (np.count_nonzero(image_ROI[i, :])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_row_array.append(i)
                
                if (np.count_nonzero(image_ROI[:, j])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_col_array.append(j)

        # remove zeros to get our ROI
        image_ROI = np.delete(image_ROI, tuple(zero_row_array), axis = 0)
        image_ROI = np.delete(image_ROI, tuple(zero_col_array), axis = 1)
   
        # preserve aspect ratio by setting both dimensions to the higher dimension
        temp_img = Image.fromarray(image_ROI, 'L')
        temp_img.save('images/(c)_ROI_Extraction.png')

        pixel_width, pixel_height = temp_img.size
        new_aspect_ratio = max(pixel_width, pixel_height)
        image_ROI = temp_img.resize((new_aspect_ratio, new_aspect_ratio))

        # Resize image to 26 by 26 to add a 2 pixel border
        image_ROI = image_ROI.resize((26,26))

        # Create a blank 28,28 black image
        image_centered = Image.new('L', (28,28))

        # Paste the 20,20 in the center to make the completed 28,28
        image_centered.paste(image_ROI, (1,1))
        image_centered.save('images/(d)_Centered_Frame.png')

        # flip and rotate 90 degrees
        newImg_flip = ImageOps.mirror(image_centered)
        newImg_rotate = newImg_flip.rotate(90)

        # final processed output image
        newImg_rotate.save('./images/(e)_Resized.png')

        # adjust for correct dtype
        image_adjust = np.array(newImg_rotate).astype(np.float32) / 255
        image_adjust_Tensor = torch.from_numpy(image_adjust)
        
        # convert 2D tensor to a 4D input by adding two dimensions for batch loading
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0)
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0)

        # feed to model
        self.output = self.model(torch.Tensor(image_adjust_Tensor))
        #print(self.output)

        # find element with maximum value
        max_prob_prediction = torch.argmax(self.output)

        # set up to find accuracy
        # normalize between 0 and 1
        accuracy = F.softmax(self.output, dim = 1)
        accuracy_numpy = accuracy.detach().numpy()
        #print(accuracy)
        #print(accuracy_numpy)
        accuracy_numpy_100 = accuracy_numpy*100     # set as percentage
        accuracy_rounded = np.round_(accuracy_numpy_100, decimals = 1)  # round to 1 dp
        #print(accuracy_rounded)
        accuracy_final = str(np.amax(accuracy_rounded))
        #print(accuracy_final)

        # set up character array to match classes of dataset
        characters_array = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                        'a','b','d','e','f','g','h','n','q','r','t']
        index = int(max_prob_prediction.item())
        predicted_character = characters_array[index]
        #print(predicted_character)

        return predicted_character, accuracy_final



# Adapted from COMPSYS 302 PyTorch Lab, set this as Default_Net
class Default_Net(nn.Module):
    def __init__(self):
        super(Default_Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 47)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


# Adapted from https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# Set this as CNN_Net
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.convolutional_layers = nn.Sequential(
            # Define first convolutional layer
            nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),

            # Apply activation function to make the network non-linear
            nn.ReLU(),   

            # Apply a 2D max pooling over input signal               
            nn.MaxPool2d(kernel_size = 2),  

            # Define second convolutional layer 
            nn.Conv2d(16, 32, 5, 1, 2),
             # with dropout to prevent overfitting
            nn.Dropout(p = 0.2),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size = 2)  
        )
 
        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 400),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(400, 80),
            nn.ReLU(),
            nn.Linear(80, 47),
         )
 
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.linear_layers(x)
        return x
  

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # load pretrained resnet model 
        self.model = models.resnet50(pretrained = True)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 47)
    
    def forward(self, x):
        return self.model(x)
