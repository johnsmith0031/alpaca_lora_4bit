import os
class Finetune4bConfig:
    """Config holder for LLaMA 4bit finetuning
    """
    def __init__(self, dataset: str, ds_type: str,
                 lora_out_dir: str, lora_apply_dir: str, resume_checkpoint: str,
                 llama_q4_config_dir: str, llama_q4_model: str,
                 mbatch_size: int, batch_size: int,
                 epochs: int, lr: float,
                 cutoff_len: int,
                 lora_r: int, lora_alpha: int, lora_dropout: float,
                 val_set_size: float,
                 gradient_checkpointing: bool,
                 gradient_checkpointing_ratio: float,
                 warmup_steps: int, save_steps: int, save_total_limit: int, logging_steps: int,
                 checkpoint: bool, skip: bool, verbose: bool,
                 txt_row_thd: int, use_eos_token: bool, groupsize: int,
                 local_rank: int,
                 ):
        """
        Args:
            dataset (str): Path to dataset file
            ds_type (str): Dataset structure format
            lora_out_dir (str): Directory to place new LoRA
            lora_apply_dir (str): Path to directory from which LoRA has to be applied before training
            resume_checkpoint (str): Path to Specified checkpoint you want to resume.
            llama_q4_config_dir (str): Path to the config.json, tokenizer_config.json, etc
            llama_q4_model (str): Path to the quantized model in huggingface format
            mbatch_size (int): Micro-batch size
            batch_size (int): Batch size
            epochs (int): Epochs
            lr (float): Learning rate
            cutoff_len (int): Cutoff length
            lora_r (int): LoRA R
            lora_alpha (int): LoRA Alpha
            lora_dropout (float): LoRA Dropout
            gradient_checkpointing (bool) : Use gradient checkpointing
            gradient_checkpointing_ratio (float) : Gradient checkpoint ratio
            val_set_size (int): Validation set size
            warmup_steps (int): Warmup steps before training
            save_steps (int): Save steps
            save_total_limit (int): Save total limit
            logging_steps (int): Logging steps
            checkpoint (bool): Produce checkpoint instead of LoRA
            skip (bool): Don't train model
            verbose (bool): If output log of training
            txt_row_thd (int): Custom row thd for txt file
            use_eos_token (bool): Use Eos token instead of padding with 0
            groupsize (int): Group size of V2 model, use -1 to load V1 model
            local_rank (int): local rank if using torch.distributed.launch
        """
        self.dataset = dataset
        self.ds_type = ds_type
        self.lora_out_dir = lora_out_dir
        self.lora_apply_dir = lora_apply_dir
        self.resume_checkpoint = resume_checkpoint
        self.llama_q4_config_dir = llama_q4_config_dir
        self.llama_q4_model = llama_q4_model
        self.mbatch_size = mbatch_size
        self.batch_size = batch_size
        self.gradient_accumulation_steps = self.batch_size // self.mbatch_size
        self.epochs = epochs
        self.lr = lr
        self.cutoff_len = cutoff_len
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = 0 if gradient_checkpointing else lora_dropout # should be 0 if gradient checkpointing is on
        self.val_set_size = int(val_set_size) if val_set_size > 1.0 else float(val_set_size)
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.logging_steps = logging_steps
        self.checkpoint = checkpoint
        self.skip = skip
        self.verbose = verbose
        self.txt_row_thd = txt_row_thd
        self.use_eos_token = use_eos_token
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        self.ddp = self.world_size != 1
        self.device_map = "auto" if not self.ddp else {"": self.local_rank}
        if self.ddp:
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.world_size
        self.groupsize = groupsize


    def __str__(self) -> str:
        s = f"\nParameters:\n{'config':-^20}\n{self.dataset=}\n{self.ds_type=}\n{self.lora_out_dir=}\n{self.lora_apply_dir=}\n{self.llama_q4_config_dir=}\n{self.llama_q4_model=}\n\n" +\
        f"{'training':-^20}\n" +\
        f"{self.mbatch_size=}\n{self.batch_size=}\n{self.gradient_accumulation_steps=}\n{self.epochs=}\n{self.lr=}\n{self.cutoff_len=}\n" +\
        f"{self.lora_r=}\n{self.lora_alpha=}\n{self.lora_dropout=}\n{self.val_set_size=}\n" +\
        f"{self.gradient_checkpointing=}\n{self.gradient_checkpointing_ratio=}\n" +\
        f"{self.warmup_steps=}\n{self.save_steps=}\n{self.save_total_limit=}\n" +\
        f"{self.logging_steps=}\n" +\
        f"{self.checkpoint=}\n{self.skip=}\n" +\
        f"{self.world_size=}\n{self.ddp=}\n{self.device_map=}"
        return s.replace("self.", "")
