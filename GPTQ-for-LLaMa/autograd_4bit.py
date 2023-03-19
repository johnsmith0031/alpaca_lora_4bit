import quant
import torch
import numpy as np
import torch.nn as nn


def matmul4bit(x, qweight, scales, zeros):
    """
    input x: (n, m)
    qweight: (j, k)
    where m == j*8
    
    perform x @ qweight
    
    return y: 
    """
    assert qweight.shape[0] * 8 == x.shape[-1]
    outshape = tuple(list(x.shape[:-1]) + [qweight.shape[1]])
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype=torch.float32, device=x.device)
    dtype = x.dtype
    x = x.float()
    quant.quant_cuda.vecquant4matmul(x, qweight, y, scales, zeros)
    y = y.to(dtype)
    return y.reshape(outshape)


def matmul4bit_transpose(x, qweight, scales, zeros):
    """
    input x: (n, m)
    qweight: (j, k)
    where m == k
    
    perform qweight @ x.T
    
    return y: 
    """
    assert qweight.shape[1] == x.shape[-1]
    outshape = tuple(list(x.shape[:-1]) + [qweight.shape[0] * 8])
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((qweight.shape[0] * 8, x.shape[0]), dtype=torch.float32, device=x.device)
    dtype = x.dtype
    x = x.float()
    quant.quant_cuda.vecquant4transposematmul(x, qweight, y, scales, zeros)
    y = y.to(dtype)
    return y.reshape(outshape)


class AutogradMatmul4bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, qweight, scales, zeros):
        ctx.save_for_backward(qweight, scales, zeros)
        output = matmul4bit(x, qweight, scales, zeros).clone()
        return output # equals to torch.matmul(x, qweight)

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, zeros = ctx.saved_tensors
        # print(grad_output.shape, A.shape, B.shape)
        
        # compute x @ qweight.T = (qweight @ x.T).T = f(x, qweight).T
        grad1 = matmul4bit_transpose(grad_output, qweight, scales, zeros)
        # grad2 = torch.matmul(x.transpose(-1, -2), grad_output)
        
        return grad1, None, None, None


# Assumes layer is perfectly divisible into 256 * 256 blocks
class Autograd4bitQuantLinear(nn.Module): 

    def __init__(self, infeatures, outfeatures):
        super().__init__()
        bits = 4
        self.in_features = infeatures
        self.out_features = outfeatures
        self.bits = bits
        self.register_buffer('zeros', torch.empty((outfeatures, 1)))
        self.register_buffer('scales', torch.empty((outfeatures, 1)))
        self.register_buffer('bias', torch.empty(outfeatures))
        self.register_buffer(
            'qweight', torch.empty((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int)
        )

    def forward(self, x):
        out = AutogradMatmul4bit.apply(x, self.qweight, self.scales, self.zeros)
        out += self.bias
        return out


def make_quant_for_4bit_autograd(module, names, name=''):
    if isinstance(module, Autograd4bitQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Autograd4bitQuantLinear(tmp.in_features, tmp.out_features)
            )
    for name1, child in module.named_children():
        make_quant_for_4bit_autograd(child, names, name + '.' + name1 if name != '' else name1)


def load_llama_model_4bit_low_ram(config_path, model_path):
    import transformers
    import accelerate
    from transformers import LLaMAConfig, LLaMAForCausalLM, LLaMATokenizer
    from modelutils import find_layers
    
    print("Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LLaMAConfig.from_pretrained(config_path)
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LLaMAForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers)
    model = accelerate.load_checkpoint_and_dispatch(model=model, checkpoint=model_path, device_map='auto')
    model.cuda()
    model.seqlen = 2048

    tokenizer = LLaMATokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    
    return model, tokenizer
    
