import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var106 = torch.nn.Conv2d(3, 64, 3, stride=3, padding=3)
        self.var107 = torch.nn.ReLU()
        self.var426 = torch.nn.Conv2d(64, 64, 3, stride=3, padding=3)
        self.var427 = torch.nn.ReLU()
        self.var431 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var870 = torch.nn.Conv2d(64, 128, 3, stride=3, padding=3)
        self.var871 = torch.nn.ReLU()
        self.var958 = torch.nn.Conv2d(128, 128, 3, stride=3, padding=3)
        self.var959 = torch.nn.ReLU()
        self.var963 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var1320 = torch.nn.Conv2d(128, 256, 3, stride=3, padding=3)
        self.var1321 = torch.nn.ReLU()
        self.var1518 = torch.nn.Conv2d(256, 256, 3, stride=3, padding=3)
        self.var1519 = torch.nn.ReLU()
        self.var1858 = torch.nn.Conv2d(256, 256, 3, stride=3, padding=3)
        self.var1859 = torch.nn.ReLU()
        self.var1863 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var2068 = torch.nn.Conv2d(256, 512, 3, stride=3, padding=3)
        self.var2069 = torch.nn.ReLU()
        self.var2219 = torch.nn.Conv2d(512, 512, 3, stride=3, padding=3)
        self.var2220 = torch.nn.ReLU()
        self.var2593 = torch.nn.Conv2d(512, 512, 3, stride=3, padding=3)
        self.var2594 = torch.nn.ReLU()
        self.var2596 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var2924 = torch.nn.Conv2d(512, 512, 3, stride=3, padding=3)
        self.var2925 = torch.nn.ReLU()
        self.var3287 = torch.nn.Conv2d(512, 512, 3, stride=3, padding=3)
        self.var3288 = torch.nn.ReLU()
        self.var3532 = torch.nn.Conv2d(512, 512, 3, stride=3, padding=3)
        self.var3533 = torch.nn.ReLU()
        self.var3545 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.var3568 = torch.nn.Flatten()
        self.var5041 = torch.nn.Linear(25088, 4096)
        self.var5042 = torch.nn.ReLU()
        self.var5595 = torch.nn.Linear(4096, 4096)
        self.var5596 = torch.nn.ReLU()
        self.var6149 = torch.nn.Linear(4096, 1000)
        self.var6152 = torch.nn.Softmax(dim=-1)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
