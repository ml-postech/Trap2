import re
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import torch


MATH_TASK_IDS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "asdiv": ("EleutherAI/asdiv", None),
}

PROMPT_TEMPLATES = {
    "gsm8k": "Question: {question}\nAnswer: Let's think step by step.\n",
    "asdiv": "Question: {question}\nAnswer: Let's think step by step.\n",
}


def _get_question_and_answer(example, task):
    """Extract question and gold answer text from a dataset example."""
    if task == "gsm8k":
        return example["question"], example["answer"]
    elif task == "asdiv":
        question = example["body"] + " " + example["question"]
        # Build CoT from formula field: "7+2=9" → "7 + 2 = 9\n#### 9"
        formula = example.get("formula", "")
        answer_raw = example.get("answer", "")
        # Extract numeric answer from "9 (apples)" format
        num_match = re.match(r'^(-?[\d.]+)', answer_raw)
        answer_num = num_match.group(1) if num_match else answer_raw
        cot = f"{formula}\n#### {answer_num}"
        return question, cot
    else:
        raise ValueError(f"Unknown math task: {task}")


def _format_prompt(question, task):
    """Format question into a prompt string."""
    if task in ("gsm8k", "asdiv"):
        return PROMPT_TEMPLATES[task].format(question=question)
    else:
        raise ValueError(f"Unknown math task: {task}")


class MathDataset:
    def __init__(self,
                 task=None,
                 model_name_or_path=None,
                 batch_size=4,
                 num_workers=8,
                 max_length=512,
                 val_fraction=0.1):

        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        hub_id, config_name = MATH_TASK_IDS[task]
        if config_name:
            raw_dataset = datasets.load_dataset(hub_id, config_name)
        else:
            raw_dataset = datasets.load_dataset(hub_id)

        # Build train/val/test splits
        val_data = None
        if "train" in raw_dataset and ("test" in raw_dataset or "validation" in raw_dataset):
            train_data = raw_dataset["train"]
            test_data = raw_dataset.get("test", raw_dataset.get("validation"))
        elif "validation" in raw_dataset and "train" not in raw_dataset:
            # Dataset has only one split (e.g., ASDiv) — split into 70/15/15
            all_data = raw_dataset["validation"]
            split1 = all_data.train_test_split(test_size=0.3, seed=42)
            train_data = split1["train"]
            remaining = split1["test"]
            split2 = remaining.train_test_split(test_size=0.5, seed=42)
            val_data = split2["train"]
            test_data = split2["test"]
        elif "train" in raw_dataset:
            train_data = raw_dataset["train"]
            test_data = None
        else:
            raise ValueError(f"No usable splits found for {task}")

        # Create val split from train if not already set
        if val_data is None:
            if val_fraction > 0:
                split = train_data.train_test_split(test_size=val_fraction, seed=42)
                train_data = split["train"]
                val_data = split["test"]

        # Build dataloaders
        self.train_loader = DataLoader(
            train_data, shuffle=True,
            collate_fn=self._train_collate_fn,
            batch_size=batch_size, num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            test_data, shuffle=False,
            collate_fn=self._eval_collate_fn,
            batch_size=batch_size, num_workers=num_workers,
        )
        if val_data is not None:
            self.val_loader = DataLoader(
                val_data, shuffle=False,
                collate_fn=self._eval_collate_fn,
                batch_size=batch_size, num_workers=num_workers,
            )
        else:
            self.val_loader = self.test_loader

    def _train_collate_fn(self, examples):
        """Collate for training: prompt + answer concatenated, labels masked on prompt."""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for ex in examples:
            question, answer = _get_question_and_answer(ex, self.task)
            prompt = _format_prompt(question, self.task)
            full_text = prompt + answer + self.tokenizer.eos_token

            tokenized = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # Find prompt length to mask in labels
            prompt_tokenized = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = prompt_tokenized["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # Left-pad to max length in batch
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, labs, mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(torch.cat([
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype), ids
            ]))
            padded_labels.append(torch.cat([
                torch.full((pad_len,), -100, dtype=labs.dtype), labs
            ]))
            padded_attention_mask.append(torch.cat([
                torch.zeros(pad_len, dtype=mask.dtype), mask
            ]))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

    def _eval_collate_fn(self, examples):
        """Collate for evaluation: prompt-only input_ids + gold answer texts."""
        prompts = []
        gold_answers = []

        for ex in examples:
            question, answer = _get_question_and_answer(ex, self.task)
            prompt = _format_prompt(question, self.task)
            prompts.append(prompt)
            gold_answers.append(answer)

        tokenized = self.tokenizer(
            prompts, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "gold_answers": gold_answers,
            "dataset_name": self.task,
        }


def prepare_train_loaders(config):
    dataset_class = MathDataset(
        task=config["type"],
        model_name_or_path=config["model_name_or_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_length=config.get("max_length", 512),
        val_fraction=config.get("val_fraction", 0.1),
    )
    return {
        "full": dataset_class.train_loader,
        "tokenizer": dataset_class.tokenizer,
    }


def prepare_test_loaders(config):
    dataset_class = MathDataset(
        task=config["type"],
        model_name_or_path=config["model_name_or_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_length=config.get("max_length", 512),
        val_fraction=config.get("val_fraction", 0.1),
    )
    return {
        "test": dataset_class.test_loader,
        "val": dataset_class.val_loader,
    }
