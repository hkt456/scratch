import torch

class ReLU(torch.autograd.Function):
    @staticmethod 
    def forward(context, input: torch.Tensor) -> torch.Tensor:
        context.save_for_backward(input)
        return torch.where(input > 0, input, torch.tensor(0.0))
    
    @staticmethod
    def backward(context, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = context.saved_tensors
        grad = torch.where(input > 0, 1, torch.tensor(0.0))
        return grad_output * grad
    
class ReLUAlternative(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.where(data < 0.0, 0.0, data)


if __name__ == "__main__":
    torch.manual_seed(0)

    relu = ReLU.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    if torch.autograd.gradcheck(relu, data, eps=1e-8, atol=1e-7):
        print("Gradient check passed")
    else:
        print("Gradient check failed")