from .base_llm import BaseLLM
from .data import Dataset, benchmark

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel
    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm

def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)
    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * question_len + input_ids[question_len:]
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100
    full["labels"] = labels
    return full

def format_example(prompt: str, answer: str) -> dict[str, str]:
    try:
        rounded_ans = f"{float(answer):.1f}"
    except ValueError:
        rounded_ans = str(answer)
    return {
        "question": f"{prompt} <answer>",
        "answer": f"{rounded_ans}</answer>"
    }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)

def train_model(output_dir: str, **kwargs):
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    llm = BaseLLM()
    model = llm.model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    if torch.cuda.is_available():
        model.enable_input_require_grads()
    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
    val_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example)
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        weight_decay=0.01,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    import os
    from pathlib import Path
    target_dir = Path(__file__).parent / "sft_model"
    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)
    print(f"Model successfully trained and saved to {target_dir}")

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=} {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
