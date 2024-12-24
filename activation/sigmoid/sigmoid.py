import torch

"""
Sigmoid activation function
In the forward pass, the sigmoid function is applied to the input tensor element-wise:
- For data > 0, the function is 1/(1+exp(-data))
- For data < 0, the function is exp(data)/(1+exp(data))

"""


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:

        negative_mask = data < 0
        positive_mask = ~negative_mask

        zs = torch.empty_like(data)
        zs[negative_mask] = data[negative_mask].exp()
        zs[positive_mask] = (-data[positive_mask]).exp()

        res = torch.ones_like(data)
        res[negative_mask] = zs[negative_mask]

        result = res/(1+zs)

        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:

        (result,) = ctx.saved_tensors
        grad = result * (1-result)

        return grad * grad_output

if __name__ == "__main__":
    torch.manual_seed(0)

    sigmoid = Sigmoid.apply
    data = torch.rand(4, dtype=torch.double, requires_grad=True)

    if torch.autograd.gradcheck(sigmoid, data, eps=1e-8, atol=1e-7):
        print("The gradient is correct!")
    else:
        print("The gradient is incorrect!")