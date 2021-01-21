import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var816 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.var817 = torch.nn.ReLU()
        self.var1528 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.var1529 = torch.nn.ReLU()
        self.var1530 = torch.nn.Flatten()
        self.var1710 = torch.nn.Linear(64, 10)
        self.var1801 = torch.nn.Linear(10, 1568)
        self.var1802 = torch.nn.ReLU()
        self.many_2 = lambda x: x.view(10, 7, 7, 32)
        self.var5089 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=0)
        self.var5090 = torch.nn.ReLU()
        self.var7498 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0)
        self.var7499 = torch.nn.ReLU()
        self.var8396 = torch.nn.ConvTranspose2d(32, 1, 3, stride=1, padding=0)
        self.var8397 = torch.nn.ReLU()
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
