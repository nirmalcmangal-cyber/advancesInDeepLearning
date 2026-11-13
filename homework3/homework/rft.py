from .base_llm import BaseLLM
from .data import Dataset, benchmark
import json
from pathlib import Path

def load() -> BaseLLM:
    from peft import PeftModel
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm

def tokenize(tokenizer, question: str, reasoning: str):
    # We train the model to output both the reasoning and the answer
    full_text = f"{question} {reasoning}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Only calculate loss on the reasoning + answer part
    labels = [-100] * question_len + input_ids[question_len:]
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

class RFTDataset:
    def __init__(self, tokenizer, data_path: str):
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, _, reasoning = self.data[idx]
        # We format it similarly to how CoT format template is expected
        formatted_q = f"""Question: {question}
Reasoning:"""
        return tokenize(self.tokenizer, formatted_q, reasoning)

def train_model(output_dir: str = "homework/rft_model"):
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    llm = BaseLLM()
    model = llm.model

    peft_config = LoraConfig(
        r=32, # Higher rank as permitted
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    if torch.cuda.is_available():
        model.enable_input_require_grads()

    train_dataset = RFTDataset(llm.tokenizer, "data/rft.json")

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=5,
        weight_decay=0.01,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

    import os
    target_dir = Path(__file__).parent / "rft_model"
    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)
    print(f"RFT Model successfully trained and saved to {target_dir}")

if __name__ == '__main__':
    from fire import Fire
    Fire({"train": train_model, "load": load})
