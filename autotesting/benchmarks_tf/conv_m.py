import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.conv1 = torch.nn.Conv2d(1,32,5,1,padding=0)
        self.maxpool1 = torch.nn.MaxPool2d(2,2,padding=0)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(2,2,padding=0)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(5120,1024)
        self.out = torch.nn.Linear(1024,5)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
