import torch

class GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        cdf = 0.5*(1.0 + torch.erf(data/(2**0.5)))
        ctx.save_for_backward(data, cdf)
        return data * cdf
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        data, cdf = ctx.saved_tensors
        pdf_val = torch.distributions.Normal(0, 1).log_prob(data).exp()
        grad = cdf + data * pdf_val
        return grad_output * grad

if __name__ == "__main__":
    # Sets the manual seed for reproducible experiments
    torch.manual_seed(0)

    gelu = GELU.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    # `torch.autograd.gradcheck` takes a tuple of tensors as input, check if your gradient evaluated
    # with these tensors are close enough to numerical approximations and returns `True` if they all
    # verify this condition.
    if torch.autograd.gradcheck(gelu, data, eps=1e-8, atol=1e-7):
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")
