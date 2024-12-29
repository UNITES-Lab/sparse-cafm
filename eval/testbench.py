# this class should provide flexible support for evaluating
# many different models on the same benchmarks/metrics

import torch

class TestBench:
    
    def __init__(self, model:torch.nn.Module, config:dict):
        """
        We can simply pass in a config and dynamically instantiate the model.
        """
        pass
    
    def eval():
        pass