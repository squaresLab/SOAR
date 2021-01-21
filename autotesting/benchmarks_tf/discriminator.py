import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var31 = torch.nn.Conv2d(1, 64, 5, stride=2, padding=0)
        self.var40 = torch.nn.LeakyReLU()
        self.var109 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=0)
        self.var118 = torch.nn.LeakyReLU()
        self.var140 = torch.nn.Flatten()
        self.var1937 = torch.nn.Linear(8960, 1024)
        self.var2369 = torch.nn.Linear(1024, 2)
        self.var2378 = torch.nn.LeakyReLU()

        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
