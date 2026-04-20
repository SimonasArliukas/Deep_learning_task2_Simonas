import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

# 1. Setup paths
model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
adapter_path = "Best_model"
test_file = "mlx_data/valid.jsonl"
output_file = "worst_samples_comparison.json"

model, tokenizer = load(model_path, adapter_path=adapter_path)

results = []

print(f"Analyzing {test_file} for high-loss samples...")


with open(test_file, "r") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue

        sample = json.loads(line)
        full_text = sample.get("text", "")

        if not full_text:
            continue

        input_ids = mx.array(tokenizer.encode(full_text))[None]

        # Calculate Loss
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        logits = model(inputs)

        loss = nn.losses.cross_entropy(logits, targets)
        avg_loss = mx.mean(loss).item()

        results.append({
            "index": i,
            "loss": avg_loss,
            "full_content": full_text
        })

#Top 10 highest losses
results.sort(key=lambda x: x["loss"], reverse=True)
top_results = results[:10]


json_output = []

for rank, res in enumerate(top_results, 1):
    content = res['full_content']

    # Extract Prompt and Ground Truth
    parts = content.split("<|im_start|>assistant")
    prompt_only = parts[0].strip() + "\n<|im_start|>assistant\n"
    ground_truth = parts[1].replace("<|im_end|>", "").strip() if len(parts) > 1 else ""

    print(f"Generating response for Rank {rank} (Index {res['index']})...")

    # Models answer
    model_answer = generate(
        model,
        tokenizer,
        prompt=prompt_only,
        max_tokens=250,
        verbose=False
    )

    json_output.append({
        "rank": rank,
        "index": res['index'],
        "loss": round(res['loss'], 4),
        "prompt": prompt_only,
        "ground_truth": ground_truth,
        "model_answer": model_answer.strip()
    })

#Saving to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

print(f"\nSuccess! Comparison saved to: {output_file}")