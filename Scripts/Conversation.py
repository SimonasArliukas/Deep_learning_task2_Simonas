from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER_PATH = "Best_model"
SYSTEM_MSG = "You are a helpful assistant."

STOP_WORDS = ["<|im_end|>", "<|endoftext|>"]
MAX_TOKENS = 300
TEMPERATURE = 0.2
REP_PENALTY = 1.2
REP_CONTEXT = 50


print(f" Kraunamas modelis su adapteriais iš '{ADAPTER_PATH}'...")

try:
    model, tokenizer = load(
        MODEL_NAME,
        adapter_path=ADAPTER_PATH,
        tokenizer_config={"eos_token": "<|im_end|>"},
    )
    print(" Modelis ir adapteriai paruošti!")
except Exception as e:
    print(f" Klaida kraunant modelį: {e}")
    print("Įsitikinkite  kad ADAPTER_PATH aplanke yra adapterių failai.")
    exit()


sampler = make_sampler(temp=TEMPERATURE)
logits_processors = make_logits_processors(
    repetition_penalty=REP_PENALTY,
    repetition_context_size=REP_CONTEXT,
)

def generate_response(user_input: str) -> str:

    prompt = (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    full_text = ""

    for chunk in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            logits_processors=logits_processors,
    ):
        full_text += chunk.text


        for stop in STOP_WORDS:
            if stop in full_text:
                full_text = full_text.split(stop)[0]
                return full_text.strip()

    return full_text.strip()


def main():
    print("\n" + "=" * 50)
    print("AI TESTAVIMAS: TIESIOGINIS ADAPTERIŲ NAUDOJIMAS")
    print("=" * 50)
    print("(įvesk 'q' arba 'exit' kad išeitum)\n")

    while True:
        try:
            user_input = input(" Klausimas: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n Iki!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "exit"):
            print("Iki!")
            break

        print("Galvojama...")
        try:
            answer = generate_response(user_input)
            print(f"\n Atsakymas:\n{answer}\n")
        except Exception as e:
            print(f"\n Generavimo klaida: {e}\n")


if __name__ == "__main__":
    main()