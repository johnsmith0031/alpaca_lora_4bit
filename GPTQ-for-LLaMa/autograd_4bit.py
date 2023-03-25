import quant
import torch
import numpy as np
import torch.nn as nn
import time


# Global Buffer
buffer_mat_dic = {}
use_new = True
auto_switch = True
auto_switch_thd = 16


def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros((shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device)
    elif buffer_mat_dic[shape_of_qweight].device != device:
        buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(device)
    return buffer_mat_dic[shape_of_qweight]
    

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


def matmul4bit_half(x, qweight, scales, zeros):
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
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype=x.dtype, device=x.device)
    dtype = x.dtype
    quant.quant_cuda.vecquant4matmul_half(x, qweight, y, scales, zeros)
    y = y.to(dtype)
    return y.reshape(outshape)
    
    
def matmul4bit_transpose_half(x, qweight, scales, zeros):
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
    y = torch.zeros((qweight.shape[0] * 8, x.shape[0]), dtype=x.dtype, device=x.device)
    dtype = x.dtype
    quant.quant_cuda.vecquant4transposematmul_half(x, qweight, y, scales, zeros)
    y = y.to(dtype)
    return y.reshape(outshape)
    

def fast_4bit_forward(x, qweight, scales, zeros, bias):
    use_new_flag = use_new
    if auto_switch:
        if x.shape[1] > auto_switch_thd:
            use_new_flag = True
        else:
            use_new_flag = False
    if use_new_flag:
        buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
        quant.quant_cuda.vecquant4recons(qweight, buffer, scales, zeros)
        output = torch.matmul(x, buffer)
    else:
        output = matmul4bit(x, qweight, scales.float(), zeros.float())
    output += bias
    return output
    

class AutogradMatmul4bit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, qweight, scales, zeros):
        ctx.save_for_backward(qweight, scales, zeros)
        buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
        quant.quant_cuda.vecquant4recons(qweight, buffer, scales, zeros)
        output = torch.matmul(x, buffer).clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, zeros = ctx.saved_tensors
        buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
        quant.quant_cuda.vecquant4recons(qweight, buffer, scales, zeros)
        grad = torch.matmul(grad_output, buffer.T)
        return grad, None, None, None


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
        if torch.is_grad_enabled():
            out = AutogradMatmul4bit.apply(x, self.qweight, self.scales, self.zeros)
            out += self.bias
        else:
            out = fast_4bit_forward(x, self.qweight, self.scales, self.zeros, self.bias)
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


def model_to_half(model):
    model.half()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()
    print('Converted as Half.')


def model_to_float(model):
    model.float()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            m.zeros = m.zeros.float()
            m.scales = m.scales.float()
            m.bias = m.bias.float()
    print('Converted as Float.')


def load_llama_model_4bit_low_ram(config_path, model_path, half=False):
    import transformers
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    from modelutils import find_layers
    
    print("Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(config_path)
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers)
    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=model_path,
        device_map='auto',
        no_split_module_classes=["LlamaDecoderLayer"]
    )

    model.seqlen = 2048
    
    if half:
        model_to_half(model)

    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    
    return model, tokenizer
    
