import json
import torch 
from torch import nn

input_object = open('input.json')
inp = json.load(input_object)
inptensor = torch.tensor(inp['input_data'][0])


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.GELU() # approximation = false in our case

    def forward(self, x):
        return self.layer(x)

circuit = MyModel()


torch.onnx.export(circuit,               # model being run
                      inptensor,                   # model input (or a tuple for multiple inputs)
                      "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})