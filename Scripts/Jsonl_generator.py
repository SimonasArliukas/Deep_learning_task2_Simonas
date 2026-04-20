import json
import random

def convert_to_qwen_chatml(input_file, output_file):
    #Load your original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle for better training distribution
    random.shuffle(data)

    #Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            instruction = entry.get("instruction", "").strip()
            extra_input = entry.get("input", "").strip()
            output = entry.get("output", "").strip()

            # Combine instruction and input
            if extra_input:
                full_user_prompt = f"{instruction}\n{extra_input}"
            else:
                full_user_prompt = instruction

            #Constructing the chatml format
            chatml_string = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{full_user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )

            #Save as JSONL
            json_line = {"text": chatml_string}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(data)} items to {output_file}")

# Run the script
convert_to_qwen_chatml('Final_dataset.json', 'train.jsonl')