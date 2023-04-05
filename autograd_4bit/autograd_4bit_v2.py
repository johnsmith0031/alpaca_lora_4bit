from colorama import init, Fore, Back, Style
import torch
import torch.nn as nn
import time
import math
import triton
from triton_utils import matmul_248_kernel, trans_matmul_248_kernel


class AutogradMatmul4bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = torch.empty((input.shape[0], qweight.shape[1]), device='cuda', dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
        matmul_248_kernel[grid](input, qweight, output,
                                scales, qzeros, g_idx,
                                input.shape[0], qweight.shape[1], input.shape[1], bits, maxq,
                                input.stride(0), input.stride(1),
                                qweight.stride(0), qweight.stride(1),
                                output.stride(0), output.stride(1),
                                scales.stride(0), qzeros.stride(0))
        
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.input_shape, ctx.bits,ctx.maxq = input.shape,bits, maxq
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, bits, maxq = ctx.input_shape, ctx.bits, ctx.maxq
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        grade_input = None

        if ctx.needs_input_grad[0]:
            grade_input = torch.empty((input_shape[0], input_shape[1]), device='cuda', dtype=torch.float32)
            grid = lambda META: (triton.cdiv(input_shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(input_shape[1], META['BLOCK_SIZE_K']),)
            trans_matmul_248_kernel[grid](grad_output, qweight, grade_input,
                                          scales, qzeros, g_idx,
                                          input_shape[0], qweight.shape[1], input_shape[1], bits, maxq,
                                          grad_output.stride(0), grad_output.stride(1),
                                          qweight.stride(0), qweight.stride(1),
                                          grade_input.stride(0), grade_input.stride(1),
                                          scales.stride(0), qzeros.stride(0))
        return grade_input, None, None, None, None, None, None


class Autograd4bitQuantLinear(nn.Module):

    def __init__(self, in_features, out_features, groupsize, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = 4  # Hardcoded 4-bits quantizations
        self.maxq = 2 ** self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else in_features
        
        self.register_buffer('qweight', torch.zeros((in_features // 32 * self.bits, out_features), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(in_features / self.groupsize), out_features // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(in_features / self.groupsize), out_features), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize  for i in range(in_features)], dtype = torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features,dtype=torch.float16))
        else:
            self.bias = None
        
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = AutogradMatmul4bit.apply(x.reshape(-1,x.shape[-1]), self.qweight, self.scales, 
                                        self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out  
        return out.reshape(out_shape)


def make_quant_for_4bit_autograd(module, names, name='', groupsize=-1):
    if isinstance(module, Autograd4bitQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Autograd4bitQuantLinear(tmp.in_features, tmp.out_features, groupsize=groupsize)
            )
    for name1, child in module.named_children():
        make_quant_for_4bit_autograd(child, names, name + '.' + name1 if name != '' else name1, groupsize=groupsize)


def model_to_half(model):
    model.half()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            m.qzeros = m.qzeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()
    print(Style.BRIGHT + Fore.YELLOW + 'Converted as Half.')


def model_to_float(model):
    model.float()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            m.qzeros = m.qzeros.float()
            m.scales = m.scales.float()
            m.bias = m.bias.float()
    print(Style.BRIGHT + Fore.YELLOW + 'Converted as Float.')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def load_llama_model_4bit_low_ram(config_path, model_path, groupsize=-1, half=False, device_map="auto", seqlen=2048):
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers, groupsize=groupsize)
    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=model_path,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"]
    )

    model.seqlen = seqlen

    if half:
        model_to_half(model)

    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = 'left'

    print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

    return model, tokenizer

def load_llama_model_4bit_low_ram_and_offload_to_cpu(config_path, model_path, lora_path=None, groupsize=-1, seqlen=2048, max_memory=None):
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    if max_memory is None:
        max_memory = {0: '24Gib', 'cpu': '48Gib'}

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers, groupsize=groupsize)
    accelerate.load_checkpoint_in_model(model, checkpoint=model_path, device_map={'': 'cpu'})

    # rotary_emb fix
    for n, m in model.named_modules():
        if 'rotary_emb' in n:
            cos_cached = m.cos_cached.clone().cpu()
            sin_cached = m.sin_cached.clone().cpu()
            break

    if lora_path is not None:
        from peft import PeftModel
        from peft.tuners.lora import Linear4bitLt
        model = PeftModel.from_pretrained(model, lora_path, device_map={'': 'cpu'}, torch_dtype=torch.float32)
        print(Style.BRIGHT + Fore.GREEN + '{} Lora Applied.'.format(lora_path))

    model.seqlen = seqlen

    print('Apply half ...')
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear) or ((lora_path is not None) and isinstance(m, Linear4bitLt)):
            m.qzeros = m.qzeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()

    print('Dispatching model ...')
    device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
    model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True, main_device=0)
    torch.cuda.empty_cache()
    print(Style.BRIGHT + Fore.YELLOW + 'Total {:.2f} Gib VRAM used.'.format(torch.cuda.memory_allocated() / 1024 / 1024))

    # rotary_emb fix
    for n, m in model.named_modules():
        if 'rotary_emb' in n:
            if getattr(m, '_hf_hook', None):
                if isinstance(m._hf_hook, accelerate.hooks.SequentialHook):
                    hooks = m._hf_hook.hooks
                else:
                    hooks = [m._hf_hook]
                for hook in hooks:
                    if hook.offload:
                        if n + '.sin_cached' not in hook.weights_map.dataset.state_dict.keys():
                            hook.weights_map.dataset.state_dict[n + '.sin_cached'] = sin_cached.clone().cpu()
                            hook.weights_map.dataset.state_dict[n + '.cos_cached'] = cos_cached.clone().cpu()

    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = 'left'

    print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

    return model, tokenizer
