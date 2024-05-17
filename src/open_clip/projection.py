import torch

class FeatureToEmbeddingProjection(torch.autograd.Function):
    '''
    CLIP uses a trainable linear projection to 
    convert multimodal features to embeddings

    This is implements the projection, but can optionally
    optimize the weights using Moore-Penrose Pseudoinverses

    the received embedding should usually be L2 normalized afterwards
    '''

    @staticmethod
    def forward(ctx, input, weight):
        "CLIP feature projections don't use a bias"
        ctx.save_for_backward(input.double(), weight.double())
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_output = grad_output.double()

        # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
        # except its much faster
        grad_weight = torch.linalg.lstsq(input, grad_output).solution

        # I * W = W_t * I_t = O_t
        # keeping the pseudoinversed matrix on the left allows us to 
        # use lstsq, which is more accurate and I think faster
        weight_t = weight.t()
        grad_input_t = torch.linalg.lstsq(weight_t, grad_output.t()).solution

        return grad_input_t.t(), grad_weight

