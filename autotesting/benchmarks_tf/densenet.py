import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var1 = torch.nn.ReLU()
        self.var184 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=0)
        self.var185 = torch.nn.ReLU()
        self.var324 = torch.nn.Conv2d(128, 32, 3, stride=1, padding=0)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
