from .. import autograd_4bit
import time
import torch
from ..autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from alpaca_lora_4bit.model_attn_mlp_patch import make_quant_attn, make_fused_mlp, inject_lora_layers
import zmq
from transformers import StoppingCriteria, StoppingCriteriaList
from io import BytesIO
import gc
import threading


def decode(output_ids, tokenizer, skip_special_tokens=True):
    if skip_special_tokens:
        reply = tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply
    else:
        return tokenizer.decode(output_ids, skip_special_tokens=False)
    

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()


# Copied from https://github.com/PygmalionAI/gradio-ui/
class _SentinelTokenStoppingCriteria(StoppingCriteria):

    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]

            for i in range(len(self.sentinel_token_ids)):
                # Can't unfold, output is still too tiny. Skip.
                if trimmed_sample.shape[-1] < self.sentinel_token_ids[i].shape[-1]:
                    continue
                for window in trimmed_sample.unfold(0, self.sentinel_token_ids[i].shape[-1], 1):
                    if torch.all(torch.eq(self.sentinel_token_ids[i][0], window)):
                        return True
        return False
    

# Copy from text-generation-webui/modules/callbacks.py
class Stream(StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False
    

class ModelServer:
    
    def __init__(self, config_path, model_path, lora_path=None, groupsize=128, is_v1_model=False, quant_attn=False, port=5555, pub_port=5556):
        self.config_path = config_path
        self.model_path = model_path
        self.lora_path = lora_path
        self.groupsize = groupsize
        self.is_v1_model = is_v1_model
        self.quant_attn = quant_attn
        self.port = port
        self.model = None
        self.tokenizer = None
        self.is_generating = False
        self.socket = None
        self.socket_pub = None
        self.pub_port = pub_port
        self.topic = b'10001'

    def load_model(self):
        print("Loading {} ...".format(self.model_path))
        t0 = time.time()
        model, tokenizer = load_llama_model_4bit_low_ram(self.config_path, self.model_path, groupsize=self.groupsize, is_v1_model=self.is_v1_model)

        if not self.quant_attn and self.lora_path is not None:
            from peft import PeftModel
            from ..monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
            replace_peft_model_with_int4_lora_model()
            model = PeftModel.from_pretrained(model, self.lora_path, device_map={'': 0}, torch_dtype=torch.float16)
            print('{} Lora Applied.'.format(self.lora_path))

        print('Apply half ...')
        model.half()
        for n, m in model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear):
                if m.is_v1_model:
                    m.zeros = m.zeros.half()
                m.scales = m.scales.half()
                m.bias = m.bias.half()
        torch.cuda.empty_cache()
        print('Total {:.2f} GiB VRAM used.'.format(torch.cuda.memory_allocated() / 1024 / 1024))

        if not self.quant_attn and self.lora_path is not None:
            from ..amp_wrapper import AMPWrapper
            wrapper = AMPWrapper(model)
            wrapper.apply_generate()
            print('AMP applied.')

        if self.quant_attn:
            make_quant_attn(model, is_v1_model=self.is_v1_model)
            make_fused_mlp(model, is_v1_model=self.is_v1_model)
            print('Quantized attention applied.')

            if self.lora_path is not None:
                inject_lora_layers(model, self.lora_path, device='cuda', dtype=torch.float16)
        
        self.model, self.tokenizer = model, tokenizer
        print("Loaded in {:.2f} seconds.".format(time.time() - t0))

    def wrap_result(self, result):
        with BytesIO() as bio:
            torch.save(result, bio)
            return bio.getvalue()
    
    def unwrap_result(self, result):
        with BytesIO(result) as bio:
            return torch.load(bio, map_location='cuda')
    
    def send_generate_end_flag(self):
        data = {
            'type': 'generate_end'
        }
        self.socket_pub.send(self.topic + self.wrap_result(data))

    def generate_thread(self, *args, **kwargs):
        clear_torch_cache()
        self.is_generating = True
        try:
            self.model.generate(*args, **kwargs)
        except ValueError:
            pass
        finally:
            self.is_generating = False
            self.send_generate_end_flag()
            clear_torch_cache()

    def stop_generate(self):
        self.is_generating = False

    def run(self):
        self.load_model()
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:{}".format(self.port))
        self.socket = socket
        context_pub = zmq.Context()
        socket_pub = context_pub.socket(zmq.PUB)
        socket_pub.bind("tcp://*:{}".format(self.pub_port))
        self.socket_pub = socket_pub
        print('Server started at port {} and {}.'.format(self.port, self.pub_port))
        '''
            Message Format:
            {'function': 'generate',
             'args': ...,
             'kwargs': ...}
        '''
        while True:
            try:
                #  Wait for next request from client
                message = socket.recv()
                message = self.unwrap_result(message)
                function = message['function']
                if function == 'generate':
                    if not self.is_generating:
                        self.is_generating = True
                        args = message['args']
                        kwargs = message['kwargs']
                        input_ids = kwargs['inputs']
                        def func(x):
                            if not self.is_generating:
                                raise ValueError
                            new_tokens = len(x) - len(input_ids[0])
                            result = decode(x[-new_tokens:], self.tokenizer, True)
                            data = {
                                'type': 'generate',
                                'data': result
                            }
                            socket_pub.send(self.topic + self.wrap_result(data))
                        kwargs['stopping_criteria'] = StoppingCriteriaList([Stream(callback_func=func)])
                        t = threading.Thread(target=self.generate_thread, args=args, kwargs=kwargs)
                        t.setDaemon(True)
                        t.start()
                        socket.send(self.wrap_result({'type': 'generate_rsp', 'data': 'ok'}))
                    else:
                        print('Already generating.')
                        socket.send(self.wrap_result({'type': 'generate_rsp', 'data': 'already generating'}))
                elif function == 'stop_generate':
                    self.stop_generate()
                    socket.send(self.wrap_result({'type': 'stop_generate_rsp', 'data': 'ok'}))
                elif function == 'test':
                    print('test ok.')
                    self.socket.send(self.wrap_result(
                        {
                            'type': 'test',
                            'data': 'test ok.'
                        }
                    ))
                elif function == 'exit':
                    socket.send(self.wrap_result({'type': 'exit_rsp', 'data': 'ok'}))
                    break
                else:
                    socket.send(self.wrap_result({'type': 'rsp', 'data': 'no function'}))
                    raise ValueError('Unknown function {}'.format(function))
            except Exception as e:
                print(str(e))
                raise
        print('Server stopped.')


class ModelClient:
    
    def __init__(self, port=5555, port_sub=5556):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:{}".format(self.port))
        self.socket_sub = self.context.socket(zmq.SUB)
        self.topic = b'10001'
        self.socket_sub.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket_sub.connect("tcp://localhost:{}".format(port_sub))
        self.callback_func = None

    def wrap_result(self, result):
        with BytesIO() as bio:
            torch.save(result, bio)
            return bio.getvalue()
    
    def unwrap_result(self, result):
        with BytesIO(result) as bio:
            return torch.load(bio, map_location='cuda')
    
    def recieve_thread(self):
        while True:
            message = self.socket_sub.recv()
            message = message[len(self.topic):]
            message = self.unwrap_result(message)
            if message['type'] == 'generate':
                if self.callback_func is not None:
                    self.callback_func(message['data'], is_end=False)
            elif message['type'] == 'generate_end':
                if self.callback_func is not None:
                    self.callback_func(None, is_end=True)
                break
            else:
                print(message)
                break
        print('receive completed.')

    def start_recieving(self):
        t = threading.Thread(target=self.recieve_thread)
        t.setDaemon(True)
        t.start()

    def generate(self, *args, **kwargs):
        data = {
            'function': 'generate',
            'args': args,
            'kwargs': kwargs
        }
        self.socket.send(self.wrap_result(data))
        result = self.socket.recv()
        return result

    def stop(self):
        data = {
            'function': 'stop_generate'
        }
        self.socket.send(self.wrap_result(data))
        result = self.socket.recv()
        return result

    def test(self):
        data = {
            'function': 'test'
        }
        self.socket.send(self.wrap_result(data))
        result = self.socket.recv()
        return result
