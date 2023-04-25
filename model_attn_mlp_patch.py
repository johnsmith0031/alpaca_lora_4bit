import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaMLP
from autograd_4bit import Autograd4bitQuantLinear
import matmul_utils_4bit
import re
import json
import types


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,hidden_size,num_heads,qkv_proj,o_proj,rotary_emb,):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"f" and `num_heads`: {num_heads}).")
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,hidden_states,past_key_value = None,attention_mask = None,position_ids = None, output_attentions = False,use_cache= False):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        is_causal = past_key_value is None
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = F.scaled_dot_product_attention(query_states,key_states,value_states,is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
        

def make_quant_attn(model, is_v1_model=False):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    print('Turning off matmul cache ...')
    matmul_utils_4bit.cache_buffer = False
    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        if not is_v1_model:
            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            del q_proj.qweight
            del k_proj.qweight
            del v_proj.qweight
            qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            del q_proj.qzeros
            del k_proj.qzeros
            del v_proj.qzeros
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
            del q_proj.scales
            del k_proj.scales
            del v_proj.scales
            g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)
            del q_proj.g_idx
            del k_proj.g_idx
            del v_proj.g_idx
            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None
            if q_proj.bias is not None:
                del q_proj.bias
                del k_proj.bias
                del v_proj.bias
            torch.cuda.empty_cache()

            qkv_layer = Autograd4bitQuantLinear(in_features=q_proj.in_features,
                                                out_features=q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                                groupsize=q_proj.groupsize,
                                                is_v1_model=False)
            qkv_layer.qweight = qweights
            qkv_layer.qzeros = qzeros
            qkv_layer.scales = scales
            qkv_layer.g_idx = g_idx
            qkv_layer.bias = bias
        else:
            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            del q_proj.qweight
            del k_proj.qweight
            del v_proj.qweight
            zeros = torch.cat([q_proj.zeros, k_proj.zeros, v_proj.zeros], dim=0)
            del q_proj.zeros
            del k_proj.zeros
            del v_proj.zeros
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=0)
            del q_proj.scales
            del k_proj.scales
            del v_proj.scales
            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None
            if q_proj.bias is not None:
                del q_proj.bias
                del k_proj.bias
                del v_proj.bias
            torch.cuda.empty_cache()

            qkv_layer = Autograd4bitQuantLinear(in_features=q_proj.in_features,
                                                out_features=q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                                groupsize=-1,
                                                is_v1_model=True)
            qkv_layer.qweight = qweights
            qkv_layer.zeros = zeros
            qkv_layer.scales = scales
            qkv_layer.bias = bias

        attn = QuantLlamaAttention(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, m.rotary_emb)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        #print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)


class QuantLlamaMLP(nn.Module):
    def __init__(self, old_module, is_v1_model=False):
        super().__init__()

        gate_proj = old_module.gate_proj
        up_proj = old_module.up_proj

        if not is_v1_model:
            qweights = torch.cat([gate_proj.qweight, up_proj.qweight], dim=1)
            del gate_proj.qweight
            del up_proj.qweight
            qzeros = torch.cat([gate_proj.qzeros, up_proj.qzeros], dim=1)
            del gate_proj.qzeros
            del up_proj.qzeros
            scales = torch.cat([gate_proj.scales, up_proj.scales], dim=1)
            del gate_proj.scales
            del up_proj.scales
            g_idx = torch.cat([gate_proj.g_idx, up_proj.g_idx], dim=0)
            del gate_proj.g_idx
            del up_proj.g_idx 
            bias = torch.cat([gate_proj.bias, up_proj.bias], dim=0) if gate_proj.bias is not None else None
            if gate_proj.bias is not None:
                del gate_proj.bias
                del up_proj.bias 
            torch.cuda.empty_cache()

            self.gate_up_proj = Autograd4bitQuantLinear(in_features=gate_proj.in_features,
                                                        out_features=gate_proj.out_features + up_proj.out_features,
                                                        groupsize=gate_proj.groupsize,
                                                        is_v1_model=False)
            self.gate_up_proj.qweight = qweights
            self.gate_up_proj.qzeros = qzeros
            self.gate_up_proj.scales = scales
            self.gate_up_proj.g_idx = g_idx
            self.gate_up_proj.bias = bias
        else:
            qweights = torch.cat([gate_proj.qweight, up_proj.qweight], dim=1)
            del gate_proj.qweight
            del up_proj.qweight
            zeros = torch.cat([gate_proj.zeros, up_proj.zeros], dim=0)
            del gate_proj.zeros
            del up_proj.zeros
            scales = torch.cat([gate_proj.scales, up_proj.scales], dim=0)
            del gate_proj.scales
            del up_proj.scales
            bias = torch.cat([gate_proj.bias, up_proj.bias], dim=0) if gate_proj.bias is not None else None
            if gate_proj.bias is not None:
                del gate_proj.bias
                del up_proj.bias 
            torch.cuda.empty_cache()

            self.gate_up_proj = Autograd4bitQuantLinear(in_features=gate_proj.in_features,
                                                        out_features=gate_proj.out_features + up_proj.out_features,
                                                        groupsize=gate_proj.groupsize,
                                                        is_v1_model=True)
            self.gate_up_proj.qweight = qweights
            self.gate_up_proj.zeros = zeros
            self.gate_up_proj.scales = scales
            self.gate_up_proj.bias = bias

        self.down_proj = old_module.down_proj
        self.act_fn = old_module.act_fn
        self.intermediate_size = gate_proj.out_features

    def forward(self, x):
        intermediate = self.gate_up_proj(x)
        gate, up = torch.split(intermediate, self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
    

def make_fused_mlp(m, parent_name='', is_v1_model=False):
    """
    Replace all LlamaMLP modules with QuantLlamaMLP modules, which fuses many of the operations.
    """
    if isinstance(m, LlamaMLP):
        return QuantLlamaMLP(m, is_v1_model=is_v1_model)

    for name, child in m.named_children():
        child = make_fused_mlp(child, parent_name=f"{parent_name}.{name}", is_v1_model=is_v1_model)

        if isinstance(child, QuantLlamaMLP):
            setattr(m, name, child)	
    return m


class CustomLoraLayerMerged(torch.nn.Module):
    
    def __init__(self, lora_A, lora_B):
        super().__init__()
        self.lora_A = torch.nn.Parameter(lora_A, requires_grad=False)
        self.lora_B = torch.nn.Parameter(lora_B, requires_grad=False)
    
    def forward(self, x):
        out = torch.einsum('bjm,ndm,nkd->nbjk', x, self.lora_A, self.lora_B)
        return out
    

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
        lora_out = self.lora_layer(x)
        q, v = lora_out[0], lora_out[1]
        dim = self.module.out_features // 3
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
        
        lora_A_q = lora_weight_dic_tmp[k1].to(device=device, dtype=dtype)
        lora_B_q = lora_weight_dic_tmp[k2].to(device=device, dtype=dtype)
        lora_A_v = lora_weight_dic_tmp[k3].to(device=device, dtype=dtype)
        lora_B_v = lora_weight_dic_tmp[k4].to(device=device, dtype=dtype)

        loraA_weight = torch.concat([lora_A_q.unsqueeze(0), lora_A_v.unsqueeze(0)], dim=0)
        loraB_weight = torch.concat([lora_B_q.unsqueeze(0), lora_B_v.unsqueeze(0)], dim=0)
        loraA_weight *= scaling
        
        lora_layer = CustomLoraLayerMerged(loraA_weight, loraB_weight)
        lora_layer = lora_layer.to(device=device, dtype=dtype)
        lora_layers[prefix] = lora_layer

    # Injection
    wrappers = []
    for n, m in model.named_modules():
        if 'qkv_proj' in n and isinstance(m, Autograd4bitQuantLinear):
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
    