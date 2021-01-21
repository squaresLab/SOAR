import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var985 = torch.nn.Linear(6272, 6272)
        self.var986 = torch.nn.LeakyReLU()
        self.many_0 = lambda x: x.view(10, 7, 7, 128)
        self.var6756 = torch.nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0)
        self.var6757 = torch.nn.LeakyReLU()
        self.var11007 = torch.nn.ConvTranspose2d(64, 1, 5, stride=2, padding=0)
        self.var11012 = lambda x: torch.Tensor.tanh_(x)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
