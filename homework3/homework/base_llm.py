from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # SmolLM2 needs pad_token defined for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' # Left-padding is standard for causal batch generation
        
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        # Format prompt wrapper
        return question

    def parse_answer(self, answer: str) -> float:
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]: ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]: ...

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        from tqdm import tqdm

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM Running on Micro Batches {micro_batch_size}",
                )
                for r in self.batched_generate(
                    prompts[idx : idx + micro_batch_size], num_return_sequences, temperature
                )
            ]

        if not prompts:
            return []

        n = num_return_sequences or 1
        do_sample = temperature > 0

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=n,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        if num_return_sequences is None:
            return decoded
        # Flat list (prompt-major order) -> one list of n generations per prompt
        return [decoded[i * n : (i + 1) * n] for i in range(len(prompts))]

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)

if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})