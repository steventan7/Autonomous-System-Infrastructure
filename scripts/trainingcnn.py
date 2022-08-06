import os
import argparse
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cnnmodel import NeuralNetwork
#from numpy import load
#data  = load("imagearray1.npy")
#data1 = load("twistarray1.npy")
#print(data1)
from loader import drivingData
from torch.utils.data.sampler import SubsetRandomSampler

#image_size = 424*240*3

def traincnn(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        #instantiate loss function - MSE
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if True:
            print("\n batch no. {}: {}/{} samples \t Loss: {:.6f}".format(epoch,
                batch_idx * len(data), len(train_loader.dataset), 
                100.0 * batch_idx / len(train_loader), loss.item()))

def testingcnn(model, dev, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(dev), target.to(dev)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("epoch {}: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100.0 * correct/len(test_loader.dataset)))

#----------------------------------------------------------------

def main():
    #The ArgumentParser object will hold all the information...
    #...necessary to parse the command line into Python data types.
    parser = argparse.ArgumentParser(description="Create Dataset from directory")
    #parsing command-line arguments
    parser.add_argument("epochs", default=2, help="number of epochs")
    parser.add_argument("data_dir", help="Directory of npz files")
    parser.add_argument("weights_path", help="Directory of the weights after training", default="")
    #parse_args will take the arguments you provide on the command line when you run your program 
    #and interpret them according to the arguments you have added to your ArgumentParser object.
    args = parser.parse_args()

    dataset = drivingData(args.data_dir)
    ntrain = len(dataset)
    indices = list(range(ntrain))
    train_sampler = SubsetRandomSampler(indices)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=train_sampler, num_workers=4)

    #dev = torch.device('cuda:0')
    lr = 0.01
    model = NeuralNetwork().to(torch.float)
    #.to(device=dev)
    #INSTANTIATE OPTIMIZER CLASS -- SGD
    optimizer = optim.SGD(model.parameters(), lr = lr)

    for epoch in range(1, int(args.epochs) + 1):
        traincnn(model, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), os.path.join(args.weights_path, "weights.pt"))
    print(os.path.join(args.weights_path, "weights.pt"))

if __name__ == '__main__':
    main()

print("Done!")

'''
#INSTANTIATE LOSS CLASS
loss_fn = nn.CrossEntropyLoss()

#INSTANTIATE OPTIMIZER CLASS -- SGD
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#TRAIN THE MODEL
#error metric -- MSE for example
#training data through iteration --- in batches of dataset

def CNN_train(loss_fn, optimizer):
    epochs = 10
    training_acc = []
    training_loss = []
    testing_acc = []
    testing_loss = []

    for epoch in range(epochs):
        train_acc = 0.0
        train_loss = 0.0

        model.train()


CNN_train(loss_fn, optimizer)



'''
