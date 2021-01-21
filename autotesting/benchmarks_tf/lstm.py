import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var67 = torch.nn.Conv2d(1, 16, 5, stride=1, padding=0)
        self.var68 = torch.nn.ReLU()
        self.var69 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.var416 = torch.nn.Conv2d(16, 36, (5, 5), stride=1, padding=0)
        self.var417 = torch.nn.ReLU()
        self.var428 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.var447 = torch.nn.Flatten()
        self.var1237 = torch.nn.Linear(576, 128)
        self.var1238 = torch.nn.ReLU()
        self.var1456 = torch.nn.Linear(128, 10)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
