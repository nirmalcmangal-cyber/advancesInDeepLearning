from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Construct a rich dialogue with rules, a brief instruction, concise instruction, and an example.
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and extremely concise mathematical reasoning assistant. Solve the user's problem step by step, then format your final numeric answer inside <answer>number</answer> tags. Do not output anything else after the closing tag."
            },
            {
                "role": "user",
                "content": "Question: If John has 3 apples and eats 1, how many does he have left?"
            },
            {
                "role": "assistant",
                "content": "John starts with 3 apples. He eats 1 apple. 3 - 1 = 2.0. <answer>2.0</answer>"
            },
            {
                "role": "user",
                "content": f"Question: {question}"
            }
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    print("--- SAMPLE FORMATTED PROMPT ---")
    print(model.format_prompt(testset[0][0]))
    print("--------------------------------")
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == '__main__':
    from fire import Fire

    Fire({"test": test_model, "load": load})
