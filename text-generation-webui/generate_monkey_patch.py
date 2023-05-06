import modules.text_generation
from modules.text_generation import *
from alpaca_lora_4bit.server import _SentinelTokenStoppingCriteria

def get_reply_from_output_str(reply, original_question):
    if type(shared.tokenizer) is transformers.LlamaTokenizer:
        if len(original_question) > 0 and original_question[-1] not in [' ', '\n']:
            reply = ' ' + reply

    if not shared.is_chat():
        reply = original_question + apply_extensions('output', reply)

    return reply
    
def generate_reply_patched(question, original_question, seed, state, eos_token=None, stopping_strings=[]):
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']:
        generate_params[k] = state[k]

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if shared.args.no_cache:
        generate_params.update({'use_cache': False})

    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    # Add the encoded tokens to generate_params
    if shared.soft_prompt:
        inputs_embeds, filler_input_ids = generate_softprompt_input_tensors(input_ids)
        question, filler_input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, filler_input_ids, inputs_embeds)
        original_input_ids = input_ids
        generate_params.update({'inputs_embeds': inputs_embeds})
        generate_params.update({'inputs': filler_input_ids})
    else:
        question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
        original_input_ids = input_ids
        generate_params.update({'inputs': input_ids})
        if inputs_embeds is not None:
            generate_params.update({'inputs_embeds': inputs_embeds})

    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    stopping_criteria_list = transformers.StoppingCriteriaList()
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
            break

    # Update generate_params with the eos token and the stopping strings
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = stopping_criteria_list

    t0 = time.time()
    token_count = -1
    try:
        if not shared.is_chat() and shared.model_type != 'HF_seq2seq':
            yield original_question

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            if shared.soft_prompt:
                output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

            yield get_reply_from_output_ids(output, input_ids, original_question, state)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:
        
            # Repalced Original with another socket server
            from queue import Queue
            queue = Queue()
            def callback_func(x, is_end=False):
                if not is_end:
                    queue.put(x)
                else:
                    queue.put(None)

            shared.model.callback_func = callback_func
            rsp = shared.model.generate(**generate_params)
            rsp = shared.model.unwrap_result(rsp)
            if rsp['data'] != 'ok':
                raise Exception(rsp['data'])
            shared.model.start_recieving()

            token_count = 0
            while True:
                reply = queue.get()
                if reply is None:
                    break
                token_count += 1
                yield get_reply_from_output_str(reply, original_question)

    except Exception:
        traceback.print_exc()
    finally:
        clear_torch_cache()
        shared.model.stop()
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = token_count
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return
        
modules.text_generation.generate_reply_old = modules.text_generation.generate_reply_HF
modules.text_generation.generate_reply_HF = generate_reply_patched
print('Generate Patch Applied')
