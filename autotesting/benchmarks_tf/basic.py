import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.lin1 = torch.nn.Linear(50,128)
        self.lin2 = torch.nn.Linear(128,256)
        self.lin3 = torch.nn.Linear(256,10)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
