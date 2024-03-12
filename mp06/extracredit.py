import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights
import torch.nn.functional as F
import torch.nn as nn

DTYPE=torch.float32
DEVICE=torch.device("cpu")

class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.conv1 = nn.Conv2d(15, 6, 3)
        # self.pool = nn.MaxPool2d(5,5)
        self.conv2 = nn.Conv2d(16, 9, 5)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 72)
        self.fc3 = nn.Linear(72, 1)

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x = torch.flatten(x)
        x = torch.unflatten(x,0,(1000, 15, 8, 8))
        x = torch.reshape(x,(len(x),3,31,31))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


###########################################################################################
def trainmodel():
    # Well, you might want to create a model a little better than this...

    # model = NeuralNet()

    # this one achieve 0.913, with lr = 0.035
    # model = torch.nn.Sequential(
    #                         torch.nn.Conv2d(15, 17, 3),
    #                         torch.nn.ReLU(),
    #                         torch.nn.Conv2d(17, 20, 3),
    #                         torch.nn.ReLU(),
    #                         torch.nn.Flatten(),
    #                         torch.nn.Linear(320, 1))
    #                         # torch.nn.ReLU(),
    #                         # torch.nn.Linear(100, 1))
    # fc1 = torch.nn.Linear(8*8*15, 800)
    # fc2 = torch.nn.Linear(800, 200)
    # fc3 = torch.nn.Linear(200, 1)
    # model = torch.nn.Sequential(torch.nn.Flatten(1), fc1, torch.nn.ReLU(), fc2, torch.nn.ReLU(), fc3)

    model = torch.nn.Sequential(
                        torch.nn.Flatten(1),
                        torch.nn.Linear(15*8*8, 835),
                        torch.nn.ReLU(),
                        torch.nn.Linear(835, 245),
                        torch.nn.ReLU(),
                        torch.nn.Linear(245,1))
    

    # ... and if you do, this initialization might not be relevant any more ...
    # model[1].weight.data = initialize_weights()
    # model[1].bias.data = torch.zeros(1)


    loss_fn = torch.nn.MSELoss()
    # lr = 0.015
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1) 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02)
    # optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.3)
    # 0.025 is about 0.912
    # originally used 0.01

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    # for epoch in range(1):
    #     for x,y in trainloader:
    #         temp = torch.nn.Flatten(x)
    #         print(temp)
    #         break

    # print(trainloader)
    for epoch in range(2100):
        for x,y in trainloader:
            prediction = model(x)
            # print("prediction", prediction)
            loss = loss_fn(y, prediction)
            print("epoch: ", epoch, "loss: ", loss)            
            optimizer.zero_grad()       # Clear previous gradients, will see more about this later
            loss.backward()             # back propagation
            optimizer.step()              # pass # Replace this line with some code that actually does the training

    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()  