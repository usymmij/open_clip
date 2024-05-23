import torch
from torch import nn

class PseudoInverseProjectionRecomputation:
    """
    This class will add a dynamic bias to a tensor
    after a certain pytorch graph node on every iteration 
    using a hook

    This bias is calculated as B = (X^+)Y,
    where X^+ is the pseudoinverse of X
    """

    def __init__(self, weight, layer, lr=1e-3):
        """initialize recomputation

        Parameters:
        tensor: the tensor to be biased upon iteration
        """
        assert( isinstance(weight, nn.Parameter) )
        assert( isinstance(layer, nn.Module) )
        self.weight = weight
        self.lr = lr

        self.input = None
        self.output= None
        def hook(module, args, output):
            self.input = args[0]
            self.output = output
        layer.register_forward_hook(hook)


    def update_error(self, target):
        self.out_error = target - self.output

    def recompute(self):
        # i^+ @ e_o
        dW = torch.linalg.lstsq(self.input, self.out_error).solution
        dW *= self.lr

        with torch.no_grad():
            self.weight += dW

class ProjectionLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        return x @ self.weight

if __name__=="__main__":
    a = torch.tensor([[1.0,2.0,0], [0,1.0,0], [0,0,1.0]])
    a = nn.Parameter(a)
    l = ProjectionLayer(a)
    
    p=PseudoInverseProjectionRecomputation(a,l, lr=1)
    print('init')
    
    x = torch.tensor([[1.0, 0.0, 2.0],[2.0,3.0,1.0]])
    y = torch.tensor([[1.0, 0.0, 2.0],[2.0,3.0,1.0]])
    y_p = l(x)

    p.update_error(y)
    print(a)
    p.recompute()
    print(a)
    

