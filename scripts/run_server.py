from server import ModelServer
import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, required=True)
    arg_parser.add_argument('--model_path', type=str, required=True)
    arg_parser.add_argument('--lora_path', type=str, default=None)
    arg_parser.add_argument('--groupsize', type=int, default=-1)
    arg_parser.add_argument('--v1', action='store_true')
    arg_parser.add_argument('--quant_attn', action='store_true')
    arg_parser.add_argument('--port', type=int, default=5555)
    arg_parser.add_argument('--pub_port', type=int, default=5556)
    args = arg_parser.parse_args()

    server = ModelServer(
        config_path=args.config_path,
        model_path=args.model_path,
        lora_path=args.lora_path,
        groupsize=args.groupsize,
        is_v1_model=args.v1,
        quant_attn=args.quant_attn,
        port=args.port,
        pub_port=args.pub_port)

    server.run()
