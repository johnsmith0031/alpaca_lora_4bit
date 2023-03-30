import torch


class AMPWrapper:
    
    def __init__(self, model, options=None):
        self.model = model
        self.options = options
        if self.options is None:
            self.options = {'enabled': True, 'device_type': 'cuda'}
        
    def autocast_forward(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_forward(*args, **kwargs)
    
    def autocast_generate(self, *args, **kwargs):
        with torch.amp.autocast(**self.options):
            return self.model.non_autocast_generate(*args, **kwargs)
    
    def apply_forward(self):
        self.model.non_autocast_forward = self.model.forward
        self.model.forward = self.autocast_forward
        
    def apply_generate(self):
        self.model.non_autocast_generate = self.model.generate
        self.model.generate = self.autocast_generate
