import torch


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''[SOURCE STRUCTURE CODE STARTS HERE]'''
        self.var36 = torch.nn.Embedding(50, 10)
        self.var162 = torch.nn.Conv1d(10, 64, 3, stride=1, padding=0)
        self.var163 = torch.nn.ReLU()
        self.var266 = torch.nn.Conv1d(64, 64, 3, stride=1, padding=0)
        self.var267 = torch.nn.ReLU()
        self.var273 = torch.nn.MaxPool1d(2, stride=3, padding=1)
        self.var276 = torch.nn.Flatten()
        self.var703 = torch.nn.Linear(192, 100)
        self.var704 = torch.nn.ReLU()
        self.var918 = torch.nn.Linear(100, 10)
        self.var920 = torch.nn.Softmax(dim=1)
        '''[SOURCE STRUCTURE CODE ENDS HERE]'''

    def forward(self, x):
        # forward pass...
        '''[SOURCE FORWARD-PASS CODE STARTS HERE]'''

        '''[SOURCE FORWARD-PASS CODE ENDS HERE]'''
        return x
