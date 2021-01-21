import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var36 = torch.nn.Embedding(50, 10)
        self.var193 = torch.nn.Conv1d(10, 16, 3, stride=1, padding=0)
        self.var286 = torch.nn.Conv1d(16, 8, 3, stride=1, padding=0)
        self.var287 = torch.nn.Flatten()
        self.var532 = torch.nn.Linear(48, 1)

        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
