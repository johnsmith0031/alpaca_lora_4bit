import math
import re
import torch
import warnings
import bitsandbytes as bnb

from peft.tuners import lora
from peft.tuners.lora import is_bnb_available, Linear, Linear8bitLt, LoraLayer
from peft.utils import _get_submodules, PeftType
from torch import nn
from transformers.pytorch_utils import Conv1D

from autograd_4bit import Autograd4bitQuantLinear


class Linear4bitLt(Autograd4bitQuantLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features,
            out_features,
            groupsize: int = -1,
            is_v1_model: bool = False,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
    ):
        Autograd4bitQuantLinear.__init__(
            self,
            in_features,
            out_features,
            groupsize,
            is_v1_model
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.qweight.requires_grad = False
            self.scales.requires_grad = False
            if self.is_v1_model:
                self.zeros.requires_grad = False
            else:
                self.qzeros.requires_grad = False
                self.g_idx.requires_grad = False
            self.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                result += output
            else:
                output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                result += output
        return result


class GPTQLoraModel(lora.LoraModel):
    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if isinstance(old_module, Autograd4bitQuantLinear) and isinstance(new_module, Linear4bitLt):
            new_module.qweight = old_module.qweight
            new_module.scales = old_module.scales
            if old_module.is_v1_model:
                new_module.zeros = old_module.zeros
            else:
                new_module.qzeros = old_module.qzeros
                new_module.g_idx = old_module.g_idx
            new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.qweight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.qweight.device)
        else:
            new_module.weight = old_module.weight
            if old_module.bias is not None:
                new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.weight.device)


def replace_peft_model_with_gptq_lora_model():
    import peft.peft_model
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
