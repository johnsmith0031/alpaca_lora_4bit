def replace_peft_model_with_int4_lora_model():
    import peft.peft_model
    from peft import PeftType
    from ..models import GPTQLoraModel
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
