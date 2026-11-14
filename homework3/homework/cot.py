from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    max_new_tokens = 50
    min_new_tokens = 5
    def format_prompt(self, question: str) -> str:
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert units. Be concise: write the conversion factor, multiply, "
                    "then give the result as <answer>number</answer>. "
                    "Facts: 1 mi=1609.344 m; 1 yd=0.9144 m; 1 ft=0.3048 m; 1 in=2.54 cm; "
                    "1 pound=453.59237 g; 1 ounce=28.349523 g; 1 ton=1 tonne=1 metric ton=1000 kg; "
                    "1 gallon=3.785412 L; 1 quart=0.946353 L; 1 pint=0.473176 L; "
                    "1 fluid ounce=29.57353 mL; 1 L=1000 mL=1000 cm^3; 1 m^3=1000 L; "
                    "1 h=3600 s; 1 day=24 h; 1 week=7 days; 1 year=365.2422 days; "
                    "1 month=30.4369 days; 1 decade=10 years; 1 century=100 years; "
                    "1 byte=8 bit; 1 KB=1000 byte; 1 MB=1000 KB; 1 GB=1000 MB."
                ),
            },
            {"role": "user", "content": "How many in are there per 8 ft?"},
            {"role": "assistant", "content": "1 ft = 12 in. 8 * 12 = 96. <answer>96</answer>"},
            {"role": "user", "content": "Convert 6 mi/h into ft/s."},
            {
                "role": "assistant",
                "content": "1 mi/h = 5280/3600 = 1.4667 ft/s. 6 * 1.4667 = 8.8. <answer>8.8</answer>",
            },
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
