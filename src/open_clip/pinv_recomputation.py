import torch
from torch import nn
from torch.nn.functional import normalize as l2_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PseudoInverseProjectionRecomputation:
    """
    This class will add a dynamic bias to a tensor
    after a certain pytorch graph node on every iteration 
    using a hook

    This bias is calculated as B = (X^+)Y,
    where X^+ is the pseudoinverse of X
    """

    def __init__(self, param, layer, lr=1e-3):
        """initialize recomputation

        Parameters:
        param: the tensor to be biased upon iteration
        layer: the layer to attach the hooks
        """
        assert( isinstance(param, nn.Parameter) )
        assert( isinstance(layer, ProjectionLayer) )
        self.weight = param 
        self.lr = lr

        self.input = None
        self.output= None
        # hook includes batch size!
        def hook(module, args, output):
            self.input = l2_norm(args[0], dim=-1)
            self.output = l2_norm(output, dim=-1)
        layer.register_forward_hook(hook)

        self.identity_pair_constant = None

    def update_error(self, target):
        if self.identity_pair_constant is None or not self.identity_pair_constant.size() == self.input.size():
            self.identity_pair_constant = (2*torch.eye(target.size()[0], target.size()[1]) - 1).to(device)

        # shapes = (batch_size, feature_dims) 
        self.out_error = l2_norm(target, dim=-1) - self.output
        self.out_error *= self.identity_pair_constant

    def recompute(self):
        # i^+ @ e_o
        dW = torch.linalg.lstsq(self.input.float(), self.out_error).solution
        dW *= self.lr

        with torch.no_grad():
            self.weight += dW

class ProjectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return x @ weight

if __name__=="__main__":
    w = torch.tensor([[0.5,0.0,0], [0,1.0,0], [0,0,1.0]])
    w = nn.Parameter(w)
    l = ProjectionLayer(w)
    
    p=PseudoInverseProjectionRecomputation(w,l, lr=1)
    print('init')
    
    x = torch.tensor([[1.0, 0.0, 2.0],[2.0,3.0,1.0]])
    y = torch.tensor([[1.0, 0.0, 2.0],[2.0,3.0,1.0]])
    print('target')
    print(l2_norm(y, dim=-1))
    y_ = l(x)
    print('\norig output')
    print(l2_norm(y_, dim=-1))

    p.update_error(y)
    p.recompute()

    y_ = l(x)
    print('\nnew output')
    print(l2_norm(y_, dim=-1))

