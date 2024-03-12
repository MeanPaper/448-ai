# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


import pprint

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        
        self.images_RGB = []         # image rgb array
        self.image_labels = []       # image label array
        self.image_filenames = []    # image file names

        for data_file_name in data_files:           # open all the incoming data files
            data_dict = unpickle(data_file_name)    # unpickle data file
            self.images_RGB += data_dict[b'data']        # load data
            self.image_labels += data_dict[b'labels']    # load labels
            self.image_filenames += data_dict[b'filenames']  # load files names
        
        self.images_RGB = np.array(self.images_RGB)

        # raise NotImplementedError("You need to write this part!")
        self.transform = transform                  # assign transform
        self.target_transform = target_transform    # assign target transform

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        # raise NotImplementedError("You need to write this part!")
        return len(self.image_labels)   # return the length of the data set

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """

        # raise NotImplementedError("You need to write this part!")
        image = self.images_RGB[idx].reshape((3,32,32)) # split the array into 3 pieces, then convert to 32x32
        image = image.transpose((1, 2, 0))              # transpose the image
                
        if self.transform:
            image = self.transform(image)        
        return image, self.image_labels[idx]



    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    if mode == "train": # add a flip
        return transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # raise NotImplementedError("You need to write this part!")


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    # raise NotImplementedError("You need to write this part!")
    # print(type(data_files))
    new_data = CIFAR10(data_files, transform=transform) # CIFAR10
    return new_data


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    
    # raise NotImplementedError("You need to write this part!")
    
    loader = DataLoader(dataset=dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle']) # dataloader
    return loader



"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################

        # raise NotImplementedError("You need to write this part!")
        

        
        load_dict = torch.load('resnet18.pt')
        backbone = resnet18(pretrained=True)
        backbone.load_state_dict(load_dict)
        backbone.eval()
        # save the checkpoint load backbone
        # use resnet18, load state dict
        # export state dict ??
        self.backbone_layer = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )
        
        for param in self.backbone_layer.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_5 = nn.Linear(512, 256)
        self.fc3_5 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(512,8)
        # self.fc3 = nn.Linear(512, 2)

        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################

        x = self.backbone_layer(x)
        x = x.reshape(x.size(0), -1)
        y = self.fc4(x)

        # x = F.relu(self.fc2_5(x))
        # x = F.relu(self.fc3_5(x))
        # x = self.fc3(x)
        # y = x
        return y
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    # raise NotImplementedError("You need to write this part!")
    if optim_type == "Adam":
        return torch.optim.Adam(params=model_params, lr = hparams['lr'])
    return torch.optim.SGD(params=model_params, lr = hparams['lr'])


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    
    # raise NotImplementedError("You need to write this part!")
    current = 0
    total = 40000

    model.train()
    for features, labels in train_dataloader:
        prediction = model(features)
        loss = loss_fn(prediction, labels)
        # current += 1
        output_str = f'loss: {loss}, {current} / {total}'
        print(output_str)
        current += labels.size(0)
        optimizer.zero_grad()   # Clear previous gradients, will see more about this later
        loss.backward()   # back propagation
        optimizer.step() 
    model.eval()
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for feature, labels in test_dataloader:
            outputs = model(feature)
            _, predicted = outputs.max(1)
            total += predicted.size(0)
            correct += (predicted == labels).sum()
            # if (total==100): break
    return 100 * (correct / total)
    # test_loss = something
    # print("Test loss:", test_loss)
    # raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    # raise NotImplementedError("You need to write this part!")
 
    
    # building training dataset and training data loader
    data_set_paths = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3",
                     "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"]
    # train_data_set = build_dataset(data_set_paths, transform=transforms.ToTensor()) 
    train_data_set = build_dataset(data_set_paths, transform=get_preprocess_transform('train'))
    train_loader_params = {"batch_size": 25, "shuffle": True}
    train_data_loader = build_dataloader(train_data_set, loader_params=train_loader_params)


    # building test dataset and test data loader
    test_set_paths = ["cifar10_batches/test_batch"]
    test_data_set = build_dataset(test_set_paths, transform=transforms.ToTensor())
    test_loader_params = {"batch_size": 64, "shuffle": True}
    test_data_loader = build_dataloader(test_data_set, loader_params=test_loader_params)

    # modeling and use for fine tuning
    model = build_model()    
    loss_fn = nn.CrossEntropyLoss()     #  loss function
    hyparam = {'lr': 0.015}
    optimizer = build_optimizer("Adam", model.parameters(), hyparam) # optimizer
    # epochs = 5
    # print(len(train_data_set))

    for epoch in range(1):
        print("epoch: ", epoch)
        train(train_data_loader, model, loss_fn, optimizer)  # You need to write this function.
        print("acc: ", '%.2f' % test(test_data_loader, model))  # optional, to monitor the training progress
    
    return model
