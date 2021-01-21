import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=0)
        self.conv2 = torch.nn.MaxPool2d(2, 2, 0)
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.softmax(x)
        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
