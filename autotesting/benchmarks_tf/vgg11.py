import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var106 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.var107 = torch.nn.ReLU()
        self.var111 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var217 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.var218 = torch.nn.ReLU()
        self.var220 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var335 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.var336 = torch.nn.ReLU()
        self.var519 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.var520 = torch.nn.ReLU()
        self.var523 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.var828 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.var829 = torch.nn.ReLU()
        self.var1143 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var1144 = torch.nn.ReLU()
        self.var1148 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.var1240 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var1241 = torch.nn.ReLU()
        self.var1388 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var1389 = torch.nn.ReLU()
        self.var1401 = torch.nn.MaxPool2d((2, 2), stride=2, padding=1)
        self.var1444 = torch.nn.Flatten()
        self.var2909 = torch.nn.Linear(25088, 4096)
        self.var2910 = torch.nn.ReLU()
        self.var3351 = torch.nn.Linear(4096, 4096)
        self.var3352 = torch.nn.ReLU()
        self.var3793 = torch.nn.Linear(4096, 1000)
        self.var3796 = torch.nn.Softmax(dim=None)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
