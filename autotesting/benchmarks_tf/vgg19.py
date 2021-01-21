import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var106 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.var107 = torch.nn.ReLU()
        self.var426 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.var427 = torch.nn.ReLU()
        self.var431 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var870 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.var871 = torch.nn.ReLU()
        self.var958 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.var959 = torch.nn.ReLU()
        self.var963 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var1320 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.var1321 = torch.nn.ReLU()
        self.var1518 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.var1519 = torch.nn.ReLU()
        self.var1858 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.var1859 = torch.nn.ReLU()
        self.var2146 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.var2147 = torch.nn.ReLU()
        self.var2151 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var2207 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.var2208 = torch.nn.ReLU()
        self.var2331 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.var2332 = torch.nn.ReLU()
        self.var2436 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var2437 = torch.nn.ReLU()
        self.var2563 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var2564 = torch.nn.ReLU()
        self.var2568 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var2820 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var2821 = torch.nn.ReLU()
        self.var3034 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var3035 = torch.nn.ReLU()
        self.var3274 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var3275 = torch.nn.ReLU()
        self.var3401 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.var3402 = torch.nn.ReLU()
        self.var3415 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var3436 = torch.nn.Flatten()
        self.var5992 = torch.nn.Linear(18432, 4096)
        self.var5993 = torch.nn.ReLU()
        self.var6459 = torch.nn.Linear(4096, 4096)
        self.var6460 = torch.nn.ReLU()
        self.var6909 = torch.nn.Linear(4096, 1000)
        self.var6911 = torch.nn.Softmax(dim=-1)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
