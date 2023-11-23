#!/usr/bin/env python
'''
This example demonstrates a simple use of pycallgraph.
'''
import torch
import torch.nn.functional as F
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput



class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output =F.to_mont(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output
        return grad_input

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # In the model, use the custom function directly, not with apply
        self.custom_function = MyCustomFunction.apply

    def forward(self, x,x2):
        # In the forward pass, use the custom function directly
        x = self.custom_function(x)
        x2=F.to_mont(x2)
        result=F.add_mont(x,x2)
        result=F.sub_mont(result,x2)
        return result

# Create model instance
model = MyModel()

# Create input tensor
x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)

x2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)
# Forward pass

def main():
    graphviz = GraphvizOutput()
    graphviz.output_file = 'test/11_11/pic/basic1.png'

    with PyCallGraph(output=graphviz):
        res = model(x1,x2)
        
if __name__ == "__main__":
    main()