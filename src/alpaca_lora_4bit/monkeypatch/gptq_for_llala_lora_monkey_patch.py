import torch
import re
import json
from quant.quant_linear import QuantLinear # from GPTQ FOR LLAMA
import types


class CustomLoraLayerMerged(torch.nn.Module):
    
    def __init__(self, scaling, lora_A_q, lora_B_q, lora_A_v, lora_B_v):
        super().__init__()
        self.lora_A_q = lora_A_q
        self.lora_B_q = lora_B_q
        self.lora_A_v = lora_A_v
        self.lora_B_v = lora_B_v
        self.scaling = scaling
    
    def forward(self, x):
        q = self.lora_B_q(self.lora_A_q(x)) * self.scaling
        v = self.lora_B_v(self.lora_A_v(x)) * self.scaling
        return q, v


class LoraInjectionWrapper:

    def __init__(self, module, lora_layer):
        self.module = module
        self.lora_layer = lora_layer

    def apply(self):
        self.module.forward_before_lora = self.module.forward
        self.module.forward = self.forward_with_lora
        self.module.is_lora_injected = True

    def forward_with_lora(self, x):
        result = self.module.forward_before_lora(x)
        q, v = self.lora_layer(x)
        dim = self.module.outfeatures // 3
        result[:, :, :dim] += q
        result[:, :, -dim:] += v
        return result
    

def inject_lora_layers(model, lora_path, device='cuda', dtype=torch.float16):

    print('Device: {}, dtype: {}'.format(device, dtype))

    with open(lora_path + '/adapter_config.json', 'r') as file:
        lora_config = json.load(file)
    scaling = lora_config['lora_alpha'] / lora_config['r']

    lora_weight_dic = {}
    dic = torch.load(lora_path + '/adapter_model.bin')
    for k, v in dic.items():
        k_new = k.replace('base_model.model.', '')
        prefix = re.findall('^model\.layers\.\d+\.', k_new)[0]
        k_new = k_new.replace(prefix, '')
        if prefix not in lora_weight_dic.keys():
            lora_weight_dic[prefix] = {}
        lora_weight_dic[prefix][k_new] = v

    lora_layers = {}
    for prefix, lora_weight_dic_tmp in lora_weight_dic.items():
        k1 = 'self_attn.q_proj.lora_A.weight'
        k2 = 'self_attn.q_proj.lora_B.weight'
        k3 = 'self_attn.v_proj.lora_A.weight'
        k4 = 'self_attn.v_proj.lora_B.weight'
        
        weight = lora_weight_dic_tmp[k1]
        l_dim = weight.shape[0]
        r_dim = weight.shape[1]
        lora_A_q = torch.nn.Linear(in_features=r_dim, out_features=l_dim, bias=False)
        lora_A_q.weight = torch.nn.Parameter(weight, requires_grad=False)
        
        weight = lora_weight_dic_tmp[k2]
        l_dim = weight.shape[0]
        r_dim = weight.shape[1]
        lora_B_q = torch.nn.Linear(in_features=r_dim, out_features=l_dim, bias=False)
        lora_B_q.weight = torch.nn.Parameter(weight, requires_grad=False)
        
        weight = lora_weight_dic_tmp[k3]
        l_dim = weight.shape[0]
        r_dim = weight.shape[1]
        lora_A_v = torch.nn.Linear(in_features=r_dim, out_features=l_dim, bias=False)
        lora_A_v.weight = torch.nn.Parameter(weight, requires_grad=False)
        
        weight = lora_weight_dic_tmp[k4]
        l_dim = weight.shape[0]
        r_dim = weight.shape[1]
        lora_B_v = torch.nn.Linear(in_features=r_dim, out_features=l_dim, bias=False)
        lora_B_v.weight = torch.nn.Parameter(weight, requires_grad=False)
        
        lora_layer = CustomLoraLayerMerged(scaling, lora_A_q, lora_B_q, lora_A_v, lora_B_v)
        lora_layer = lora_layer.to(device=device, dtype=dtype)
        lora_layers[prefix] = lora_layer

    # Injection
    wrappers = []
    for n, m in model.named_modules():
        if 'qkv_proj' in n and isinstance(m, QuantLinear):
            # restoring forward
            if hasattr(m, 'is_lora_injected') and m.is_lora_injected:
                m.forward = m.forward_before_lora
            prefix = re.findall('^model\.layers\.\d+\.', n)[0]
            lora_layer = lora_layers[prefix]
            wrapper = LoraInjectionWrapper(m, lora_layer)
            wrapper.apply()
            wrappers.append(wrapper)
    
    print('Lora Injected.')
    return wrappers
