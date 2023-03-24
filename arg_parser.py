import os
import argparse
from Finetune4bConfig import Finetune4bConfig

def parse_commandline():
    parser = argparse.ArgumentParser(
        prog=__file__.split(os.path.sep)[-1],
        description="Produce LoRA in 4bit training",
        usage="%(prog)s [config] [training]\n\nAll arguments are optional"
    )
    
    parser.add_argument("dataset", nargs="?",
        default="./dataset.json", 
        help="Path to dataset file. Default: %(default)s"
    )
    
    parser_config = parser.add_argument_group("config")
    parser_training = parser.add_argument_group("training")

    # Config args group
    parser_config.add_argument("--ds_type", choices=["txt", "alpaca"], default="alpaca", required=False,
        help="Dataset structure format. Default: %(default)s"
    )
    parser_config.add_argument("--lora_out_dir", default="alpaca_lora", required=False,
        help="Directory to place new LoRA. Default: %(default)s"
    )
    parser_config.add_argument("--lora_apply_dir", default=None, required=False,
        help="Path to directory from which LoRA has to be applied before training. Default: %(default)s"
    )
    parser_config.add_argument("--llama_q4_config_dir", default="./llama-13b-4bit/", required=False,
        help="Path to the config.json, tokenizer_config.json, etc. Default: %(default)s"
    )
    parser_config.add_argument("--llama_q4_model", default="./llama-13b-4bit.pt", required=False,
        help="Path to the quantized model in huggingface format. Default: %(default)s"
    )

    # Training args group
    parser_training.add_argument("--mbatch_size", default=1, type=int, help="Micro-batch size. Default: %(default)s")
    parser_training.add_argument("--batch_size", default=2, type=int, help="Batch size. Default: %(default)s")
    parser_training.add_argument("--epochs", default=3, type=int, help="Epochs. Default: %(default)s")
    parser_training.add_argument("--lr", default=2e-4, type=float, help="Learning rate. Default: %(default)s")
    parser_training.add_argument("--cutoff_len", default=256, type=int, help="Default: %(default)s")
    parser_training.add_argument("--lora_r", default=8, type=int, help="Default: %(default)s")
    parser_training.add_argument("--lora_alpha", default=16, type=int, help="Default: %(default)s")
    parser_training.add_argument("--lora_dropout", default=0.05, type=float, help="Default: %(default)s")
    parser_training.add_argument("--val_set_size", default=0.2, type=float, help="Validation set size. Default: %(default)s")
    parser_training.add_argument("--warmup_steps", default=50, type=int, help="Default: %(default)s")
    parser_training.add_argument("--save_steps", default=50, type=int, help="Default: %(default)s")
    parser_training.add_argument("--save_total_limit", default=3, type=int, help="Default: %(default)s")
    parser_training.add_argument("--logging_steps", default=10, type=int, help="Default: %(default)s")
    parser_training.add_argument("-c", "--checkpoint", action="store_true", help="Produce checkpoint instead of LoRA. Default: %(default)s")
    parser_training.add_argument("--skip", action="store_true", help="Don't train model. Can be useful to produce checkpoint from existing LoRA. Default: %(default)s")

    return vars(parser.parse_args())


def get_config() -> Finetune4bConfig:
    args = parse_commandline()
    return Finetune4bConfig(
        dataset=args["dataset"], 
        ds_type=args["ds_type"], 
        lora_out_dir=args["lora_out_dir"], 
        lora_apply_dir=args["lora_apply_dir"],
        llama_q4_config_dir=args["llama_q4_config_dir"],
        llama_q4_model=args["llama_q4_model"],
        mbatch_size=args["mbatch_size"],
        batch_size=args["batch_size"],
        epochs=args["epochs"], 
        lr=args["lr"],
        cutoff_len=args["cutoff_len"],
        lora_r=args["lora_r"], 
        lora_alpha=args["lora_alpha"], 
        lora_dropout=args["lora_dropout"],
        val_set_size=args["val_set_size"],
        warmup_steps=args["warmup_steps"],
        save_steps=args["save_steps"],
        save_total_limit=args["save_total_limit"],
        logging_steps=args["logging_steps"],
        checkpoint=args["checkpoint"],
        skip=args["skip"]
    )
