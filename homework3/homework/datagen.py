import json
import os
from pathlib import Path
from .cot import CoTModel
from .data import Dataset

def generate_dataset(output_path: str = "data/rft.json"):
    model = CoTModel()
    dataset = Dataset("train")
    
    rft_data = []
    questions_to_process = []
    ground_truths = []
    
    for idx in range(len(dataset)):
        question, answer = dataset[idx]
        questions_to_process.append(question)
        ground_truths.append(answer)
        
    # Batch generate multiple responses per question
    # We will query with num_return_sequences=10 and temperature=0.7 to get diverse rollouts
    prompts = [model.format_prompt(q) for q in questions_to_process]
    
    print(f"Generating rollouts for {len(prompts)} questions...")
    
    # We can batch process or loop since we want to find correct CoT reasoning paths
    for i, (q, gt) in enumerate(zip(questions_to_process, ground_truths)):
        prompt = model.format_prompt(q)
        # Generate 10 candidate responses
        candidates = model.batched_generate([prompt] * 10, temperature=0.7)
        
        found_correct = False
        for cand in candidates:
            parsed = model.parse_answer(cand)
            # If candidate answer matches the ground truth, save it
            try:
                if abs(parsed - float(gt)) < 1e-4:
                    # Format as [question, ground_truth, reasoning]
                    # We extract the clean reasoning path leading to the <answer>
                    rft_data.append([q, gt, cand.strip()])
                    found_correct = True
                    break
            except (ValueError, TypeError):
                continue
                
        if found_correct:
            print(f"[Success] Question {i}: Found correct reasoning path.")
        else:
            print(f"[Skip] Question {i}: No correct reasoning path generated.")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rft_data, f, indent=2)
    print(f"RFT dataset generated with {len(rft_data)} examples and saved to {output_path}!")

if __name__ == '__main__':
    from fire import Fire
    Fire({"generate": generate_dataset})
