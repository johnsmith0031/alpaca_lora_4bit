import modules.text_generation
from modules.text_generation import *
from server import _SentinelTokenStoppingCriteria

def generate_reply_patched(question, state, eos_token=None, stopping_strings=[]):
    if shared.model_name == 'None' or shared.model is None:
        print("No model is loaded! Select one in the Model tab.")
        yield formatted_outputs(question, shared.model_name)
        return

    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    shared.stop_everything = False
    generate_params = get_generate_params(state)
    t0 = time.time()

    # Preparing the input
    original_question = question
    if not shared.is_chat():
        question = apply_extensions('input', question)

    # If the model is not on transformers, handle it separately and end this
    # function call earlier.
    if shared.model_type in ['rwkv', 'llamacpp']:
        if shared.args.verbose:
            print(f'\n\n{question}\n--------------------\n')

        try:
            if shared.args.no_stream:
                reply = shared.model.generate(context=question, **generate_params)
                output = original_question + reply
                if not shared.is_chat():
                    reply = original_question + apply_extensions('output', reply)

                yield formatted_outputs(reply, shared.model_name)
            else:
                if not shared.is_chat():
                    yield formatted_outputs(question, shared.model_name)

                for reply in shared.model.generate_with_streaming(context=question, **generate_params):
                    output = original_question + reply
                    if not shared.is_chat():
                        reply = original_question + apply_extensions('output', reply)

                    yield formatted_outputs(reply, shared.model_name)

        except Exception:
            traceback.print_exc()
        finally:
            t1 = time.time()
            original_tokens = len(encode(original_question)[0])
            new_tokens = len(encode(output)[0]) - original_tokens
            print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
            return

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed, shared.args.flexgen))
    if shared.args.verbose:
        print(f'\n\n{decode(input_ids[0], state["skip_special_tokens"])}\n--------------------\n')

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    # Create the StoppingCriteriaList with the stopping strings
    stopping_criteria_list = transformers.StoppingCriteriaList()
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
            break

    # Update generate_params with the eos token and the stopping strings
    if shared.args.flexgen:
        generate_params['stop'] = eos_token_ids[-1]
    else:
        generate_params['eos_token_id'] = eos_token_ids
        generate_params['stopping_criteria'] = stopping_criteria_list

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

    try:
        # Generate the entire reply at once.
        if shared.args.no_stream:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            if shared.soft_prompt:
                output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

            new_tokens = len(output) - len(input_ids[0])
            reply = decode(output[-new_tokens:], state['skip_special_tokens'])
            if not shared.is_chat():
                reply = original_question + apply_extensions('output', reply)

            yield formatted_outputs(reply, shared.model_name)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        elif not shared.args.flexgen:
            
            # Repalced Original with another socket server
            from queue import Queue
            queue = Queue()
            def callback_func(x, is_end=False):
                if not is_end:
                    queue.put(x)
                else:
                    queue.put(None)

            shared.model.callback_func = callback_func
            shared.model.generate(**generate_params)
            shared.model.start_recieving()

            token_count = 0
            while True:
                reply = queue.get()
                if reply is None:
                    break
                token_count += 1
                yield formatted_outputs(reply, shared.model_name)

        # Stream the output naively for FlexGen since it doesn't support 'stopping_criteria'
        else:
            for i in range(state['max_new_tokens'] // 8 + 1):
                clear_torch_cache()
                with torch.no_grad():
                    output = shared.model.generate(**generate_params)[0]

                if shared.soft_prompt:
                    output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

                new_tokens = len(output) - len(original_input_ids[0])
                reply = decode(output[-new_tokens:], state['skip_special_tokens'])
                if not shared.is_chat():
                    reply = original_question + apply_extensions('output', reply)

                if np.count_nonzero(np.isin(input_ids[0], eos_token_ids)) < np.count_nonzero(np.isin(output, eos_token_ids)):
                    break

                yield formatted_outputs(reply, shared.model_name)
                input_ids = np.reshape(output, (1, output.shape[0]))
                if shared.soft_prompt:
                    inputs_embeds, filler_input_ids = generate_softprompt_input_tensors(input_ids)
                    generate_params.update({'inputs_embeds': inputs_embeds})
                    generate_params.update({'inputs': filler_input_ids})
                else:
                    generate_params.update({'inputs': input_ids})

            yield formatted_outputs(reply, shared.model_name)

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        try:
            shared.model.stop()
        except:
            pass
        original_tokens = len(original_input_ids[0])
        new_tokens = token_count
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return

modules.text_generation.generate_reply_old = modules.text_generation.generate_reply
modules.text_generation.generate_reply = generate_reply_patched
print('Generate Patch Applied')
