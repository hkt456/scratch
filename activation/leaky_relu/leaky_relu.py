import torch

class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor, alpha: float=1e-2) -> torch.Tensor:
        ctx.save_for_backward(data, torch.tensor(alpha).double())
        return torch.where(data<0.0, alpha*data, data)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        data, alpha = ctx.saved_tensors
        grad = torch.where(data <= 0.0, alpha, 1.0)
        return grad_output * grad
    
if __name__ == "__main__":
    torch.manual_seed(0)

    relu = LeakyReLU.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    if torch.autograd.gradcheck(relu, data, eps=1e-8, atol=1e-7):
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")

