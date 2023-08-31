def replace_peft_model_with_int4_lora_model():
    import peft.peft_model
    from peft import PeftType
    from peft.tuners.lora import LoraModel
    from ..models import GPTQLoraModel
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
    LoraModel._create_new_module = GPTQLoraModel._create_new_module
    LoraModel._replace_module = GPTQLoraModel._replace_module
    print('Repalced _create_new_module and _replace_module function')
