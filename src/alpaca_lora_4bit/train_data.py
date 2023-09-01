import torch

from abc import ABC, abstractmethod
from typing import Dict, Any
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import os


# Abstract train data loader
class ATrainData(ABC):
    """
    """
    @abstractmethod
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len: int) -> None:
        """
        Args:
            dataset (str): Path to dataset
            val_set_size (int) : Size of validation set
            tokenizer (_type_): Tokenizer
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_set_size = val_set_size
        self.cutoff_len = cutoff_len
        self.train_data = None
        self.val_data = None

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        """Tokenization method

        Args:
            prompt (str): Prompt string from dataset

        Returns:
            Dict[str, Any]: token
        """
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        """Loads dataset from file and prepares train_data property for trainer
        """
        pass


# LLaMA txt train data loader
class TrainTxt(ATrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len):
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)  # TODO: Validation size isn't used
        self.cutoff_len = cutoff_len
        self.exceed_count = 0

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            d = {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
            }
            if (
                d["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(d["input_ids"]) < self.cutoff_len
            ):
                d["input_ids"].append(self.tokenizer.eos_token_id)
                d["attention_mask"].append(1)
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            d = {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }
        if sum(d['attention_mask']) >= self.cutoff_len:
            self.exceed_count += 1
        return d

    @classmethod
    def format_new_rows(cls, rows, thd=128):
        r_b = ''
        new_rows = []
        for row in rows:
            if len(r_b) == 0:
                r_b += row
            else:
                r_b += '\n' + row
            if len(r_b) > thd:
                new_rows.append(r_b)
                r_b = ''
        if len(r_b) > thd:
            new_rows.append(r_b)
            r_b = ''
        return new_rows

    def prepare_data(self, thd=-1, use_eos_token=True, **kwargs):
        if os.path.isdir(self.dataset):
            rows = []
            for filename in os.listdir(self.dataset):
                with open(self.dataset + filename, 'r', encoding='utf8') as file:
                    txt = file.read()
                txt = txt.replace('\r\n', '\n').replace('\u3000', ' ')
                rows += [r for r in txt.split('\n') if r != '']
        else:
            with open(self.dataset, 'r', encoding='utf8') as file:
                txt = file.read()
            txt = txt.replace('\r\n', '\n')
            rows = [r for r in txt.split('\n') if r != '']
        if thd != -1:
            rows = self.format_new_rows(rows, thd=thd)
        data = Dataset.from_dict({"input": rows})
        data = data.shuffle().map(lambda x: self.tokenize(x["input"], use_eos_token=use_eos_token))
        print('Train Data: {:.2f}%'.format(self.exceed_count / len(data) * 100), 'outliers')
        self.train_data = data


# Stanford Alpaca-like Data
class TrainSAD(ATrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        else:
            self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = None

    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        prompt += f"### Instruction:\n{data_point['instruction']}\n\n"
        if 'input' in data_point:
            prompt += f"### Input:\n{data_point['input']}\n\n"
        prompt += f"### Response:\n{data_point['output']}"
        return prompt

    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)
    
# Blue Moon like Data prompt-response
class TrainBlueMoon(ATrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        else:
            self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = None

    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        return "{0}\n{1}\n{2}".format(
            "prompt:",
            data_point["prompt"],
            "response:",
            data_point["response"]
        )

    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)    

# GPT4All-like Data
class TrainGPT4All(ATrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        pass

    def tokenize_inputs(self, examples):
        max_length = self.cutoff_len
        input_ids = torch.full((len(examples["prompt"]), max_length), self.tokenizer.pad_token_id)
        # ignore bos
        newline_tokens = self.tokenizer("\n", return_tensors="pt")["input_ids"][0, 1:]

        out = {"labels": [], "attention_mask": []}
        for i, (prompt, response) in enumerate(zip(examples["prompt"], examples["response"])):
            input_tokens = self.tokenizer(prompt, truncation=True, max_length=max_length // 2, return_tensors="pt")["input_ids"].squeeze()
            if input_tokens.dim() == 0:
                input_tokens = input_tokens.unsqueeze(0)

            input_len = len(input_tokens)

            # plus one since we remove bos from response
            # but we subtract one since we want to add eos token
            remaining_tokens = max_length - input_len - len(newline_tokens) + 1
            # remove bos
            target_tokens = self.tokenizer(response, truncation=True, max_length=remaining_tokens, return_tensors="pt")["input_ids"].squeeze()[1:]

            input_ids[i, :input_len] = input_tokens
            # add newline between prompt and response
            newline_plus_inputs = input_len + len(newline_tokens)
            input_ids[i, input_len: newline_plus_inputs] = newline_tokens

            # add target tokens, remove bos
            input_ids[i, newline_plus_inputs: newline_plus_inputs + len(target_tokens)] = target_tokens
            # add eos token, enforce stopping if we don't truncate
            # we don't want long code to stop generating if truncated during training
            if newline_plus_inputs + len(target_tokens) < max_length:
                input_ids[i, newline_plus_inputs + len(target_tokens)] = self.tokenizer.eos_token_id

            labels = input_ids[i].clone()
            labels[: newline_plus_inputs] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            # to debug this, can set all values == -100 to the pad token, then assert that tokenizer.decode(labels, skip_special_tokens=True).strip() == response

            attention_mask = input_ids[i].ne(self.tokenizer.pad_token_id).int()

            out["labels"].append(labels)
            out["attention_mask"].append(attention_mask)

        out["input_ids"] = input_ids

        out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}

        return out

    def prepare_data(self, **kwargs) -> None:
        dataset = load_dataset("json", data_files=self.dataset)

        self.val_data = None
        if self.val_set_size > 0:
            dataset = dataset["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            train_dataset, val_dataset = dataset["train"], dataset["test"]

            # tokenize inputs and return labels and attention mask
            val_dataset = val_dataset.map(
                lambda ele: self.tokenize_inputs(ele),
                batched=True,
                remove_columns=["source", "prompt"],
            )
            self.val_data = val_dataset.with_format("torch")
        else:
            train_dataset = dataset["train"]

        train_dataset = train_dataset.map(
            lambda ele: self.tokenize_inputs(ele),
            batched=True,
            remove_columns=["source", "prompt"],
        )
        self.train_data = train_dataset.with_format("torch")
