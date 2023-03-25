from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np


class NewForward:

    def __init__(self, layer):
        self.layer = layer
        self.apply_patch()

    def apply_patch(self):
        self.layer.old_forward_for_cp = self.layer.forward
        self.layer.forward = self.new_forward

    def new_forward(self, *args, **kwargs):
        def func(*args):
            return self.layer.old_forward_for_cp(*args, **kwargs)
        output = checkpoint(func, *args)
        return output


class VarWrapper:

    def __init__(self, model):
        self.model = model
        self.apply_patch()
        print('Var Wrapper Patch Applied')

    def apply_patch(self):
        self.model.old_forward_for_cp = self.model.forward
        self.model.forward = self.new_forward

    def new_forward(self, *args, **kwargs):
        out = self.model.old_forward_for_cp(*args, **kwargs)
        out = Variable(out.data, requires_grad=True)
        return out


def apply_gradient_checkpointing(model, checkpoint_ratio=1):
    new_forwards = []
    modules = []
    for n, m in model.named_modules():
        if isinstance(m, LlamaDecoderLayer):
            modules.append(m)
    if checkpoint_ratio < 1 and checkpoint_ratio > 0:
        checkpoint_locs = np.array((np.linspace(0, 1, int(len(modules) * checkpoint_ratio)) * (len(modules)-1)).round(), dtype=int)
    else:
        checkpoint_locs = np.arange(len(modules))
    for i in checkpoint_locs:
        m = modules[i]
        new_forwards.append(NewForward(m))
        print('Forward Patch Applied For Block {}'.format(i))
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Embedding):
            wrapper = VarWrapper(m)
            break
    return new_forwards, wrapper
