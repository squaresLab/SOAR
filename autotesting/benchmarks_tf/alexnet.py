import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var33 = torch.nn.Conv2d(3, 96, 11, 4, padding=0)
        self.var34 = torch.nn.ReLU()
        self.var40 = torch.nn.MaxPool2d(2, 2, padding=0)
        self.var875 = torch.nn.Conv2d(96, 256, 11, 1, padding=0)
        self.var876 = torch.nn.ReLU()
        self.var886 = torch.nn.MaxPool2d(2, 2, padding=0)
        self.var1408 = torch.nn.Conv2d(256, 384, 3, 1, padding=0)
        self.var1409 = torch.nn.ReLU()
        self.var1852 = torch.nn.Conv2d(384, 384, 3, 1, padding=0)
        self.var1853 = torch.nn.ReLU()
        self.var2248 = torch.nn.Conv2d(384, 256, 3, 1, padding=0)
        self.var2249 = torch.nn.ReLU()
        self.var2256 = torch.nn.MaxPool2d(3, 2, padding=1)
        self.var2258 = torch.nn.Flatten()
        self.var2508 = torch.nn.Linear(256, 4096)
        self.var2509 = torch.nn.ReLU()
        self.var3068 = torch.nn.Linear(4096, 1000)
        self.var3069 = torch.nn.ReLU()
        self.var3566 = torch.nn.Linear(1000, 17)
        self.var3569 = torch.nn.Softmax(dim=-1)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
