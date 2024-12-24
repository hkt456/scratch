import torch

class ReLUHKT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return torch.where(input > 0.0, input, 0.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad = torch.where(input > 0, 1, 0)
        return grad * grad_output    

if __name__ == "__main__":
    torch.manual_seed(0)

    relu = ReLUHKT.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    if torch.autograd.gradcheck(relu, data, eps=1e-8, atol=1e-7):
        print("Gradient check passed")
    else:
        print("Gradient check failed")

