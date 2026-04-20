import json
import os
import random
import subprocess
import textwrap
import re
import time
from pathlib import Path

#Config

MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DATASET_FILE = "Final_dataset.json"
DATA_DIR = "mlx_data"
OUTPUT_DIR = "outputs_1"
CONFIG_FILE = "lora_config.yaml"
TRAIN_SPLIT = 0.9
LOSS_LOG_FILE = "loss_log.json"

# Training Hyperparameters
ITERS = 6000
BATCH_SIZE = 1
NUM_LAYERS = 28
LEARNING_RATE = 5e-05
MAX_SEQ_LEN = 512

# Early Stopping
PATIENCE = 3
STEPS_PER_EVAL = 300
STEPS_PER_SAVE = 300


#Writing the config file
def write_config():
    config = textwrap.dedent(f"""
        model: "{MODEL_NAME}"
        train: true
        data: "{DATA_DIR}"
        iters: {ITERS}
        batch_size: {BATCH_SIZE}
        num_layers: {NUM_LAYERS}
        learning_rate: {LEARNING_RATE}
        weight_decay: 0.1
        steps_per_eval: {STEPS_PER_EVAL}
        save_every: {STEPS_PER_SAVE}
        adapter_path: "{OUTPUT_DIR}"
        max_seq_length: {MAX_SEQ_LEN}

        lora_parameters:
          rank: 32
          alpha: 64
          scale: 2.0
          dropout: 0.1
          keys:
            - "self_attn.q_proj"
            - "self_attn.k_proj"
            - "self_attn.v_proj"
            - "self_attn.o_proj"
            - "mlp.gate_proj"
            - "mlp.up_proj"
            - "mlp.down_proj"

        lr_schedule:
          name: cosine_decay
          warmup: 0
          arguments:
            - {LEARNING_RATE}
            - {ITERS}
    """).lstrip()

    with open(CONFIG_FILE, "w") as f:
        f.write(config)
    print(f"  ✅ {CONFIG_FILE} written.")


#Logging the loss
class LossLogger:
    """Logs training and validation losses to a JSON file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = {
            "config": {
                "model": MODEL_NAME,
                "iters": ITERS,
                "batch_size": BATCH_SIZE,
                "num_layers": NUM_LAYERS,
                "learning_rate": LEARNING_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "lora_rank": 32,
                "lora_alpha": 64,
            },
            "training_losses": [],  # {"step": int, "loss": float, "time": str}
            "validation_losses": [],  # {"step": int, "loss": float, "time": str}
            "best_val_loss": None,
            "stopped_early": False,
            "total_steps_run": 0,
        }
        # Load existing log if resuming
        if Path(filepath).exists():
            try:
                with open(filepath, "r") as f:
                    existing = json.load(f)
                # Preserve existing loss history on resume
                self.data["training_losses"] = existing.get("training_losses", [])
                self.data["validation_losses"] = existing.get("validation_losses", [])
                self.data["best_val_loss"] = existing.get("best_val_loss", None)
                print(f"  📂 Loaded existing loss log from {filepath} (resuming).")
            except (json.JSONDecodeError, KeyError):
                print(f"  ⚠️  Could not parse existing {filepath}. Starting fresh log.")

    def log_train(self, step: int, loss: float):
        entry = {"step": step, "loss": round(loss, 6), "time": _now()}
        self.data["training_losses"].append(entry)
        self.data["total_steps_run"] = step
        self._save()

    def log_val(self, step: int, loss: float):
        entry = {"step": step, "loss": round(loss, 6), "time": _now()}
        self.data["validation_losses"].append(entry)
        if self.data["best_val_loss"] is None or loss < self.data["best_val_loss"]:
            self.data["best_val_loss"] = round(loss, 6)
        self._save()

    def mark_early_stop(self, step: int):
        self.data["stopped_early"] = True
        self.data["total_steps_run"] = step
        self._save()

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


#Training loop
def train():
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    logger = LossLogger(LOSS_LOG_FILE)

    # Finding latest adapter file
    adapter_files = list(output_path.glob("*_adapters.safetensors"))

    latest_adapter = None
    if adapter_files:
        def extract_step(p):
            match = re.search(r'(\d+)_adapters', p.name)
            return int(match.group(1)) if match else 0

        latest_adapter = sorted(adapter_files, key=extract_step)[-1]

    # Base command
    cmd = ["python", "-m", "mlx_lm", "lora", "-c", CONFIG_FILE]

    #Resuming from the laters adapter if it exists
    if latest_adapter and latest_adapter.exists():
        print(f"🔄 Resuming from LATEST adapter file: {latest_adapter.name}")
        cmd.extend(["--resume-adapter-file", str(latest_adapter)])
    else:
        standard_file = output_path / "best_adapters.safetensors"
        if standard_file.exists():
            print(f"🔄 Resuming from standard: {standard_file.name}")
            cmd.extend(["--resume-adapter-file", str(standard_file)])
        else:
            print("🆕 No existing adapters found. Starting fresh training.")

    best_val_loss = float("inf")
    no_improve_steps = 0
    current_step = 0

    #Parsing the console to match training and validation logs for logging purposes
    train_pattern = re.compile(
        r'[Ii]ter\s+(\d+).*?[Tt]rain(?:ing)?\s+loss\s+([\d.]+)', re.IGNORECASE
    )
    val_pattern = re.compile(
        r'[Ii]ter\s+(\d+).*?[Vv]al(?:idation)?\s+loss\s+([\d.]+)', re.IGNORECASE
    )

    print(f"\n📝 Loss log will be saved to: {LOSS_LOG_FILE}")
    print("\n🚀 Starting fine-tuning on Apple Silicon...\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")

        #Parsing Training loss
        train_match = train_pattern.search(line)
        if train_match:
            step = int(train_match.group(1))
            loss = float(train_match.group(2))
            current_step = step
            logger.log_train(step, loss)

        #Parsing validation loss
        val_match = val_pattern.search(line)
        if val_match:
            step = int(val_match.group(1))
            val_loss = float(val_match.group(2))
            current_step = step
            logger.log_val(step, val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improve_steps = 0
                print(f"  ✅ New best val loss: {best_val_loss:.4f}")
            else:
                no_improve_steps += 1
                print(f"  ⚠️  No improvement ({no_improve_steps}/{PATIENCE})")

                if no_improve_steps >= PATIENCE:
                    print("\n🛑 Early stopping triggered. Terminating training.")
                    logger.mark_early_stop(current_step)
                    process.terminate()
                    break


        elif "Val loss" in line and not val_match:
            try:
                val_loss = float(
                    line.split("Val loss")[-1].strip().split()[0].rstrip(",")
                )
                logger.log_val(current_step, val_loss)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    no_improve_steps = 0
                    print(f"  ✅ New best val loss: {best_val_loss:.4f}")
                else:
                    no_improve_steps += 1
                    print(f"  ⚠️  No improvement ({no_improve_steps}/{PATIENCE})")

                    if no_improve_steps >= PATIENCE:
                        print("\n🛑 Early stopping triggered. Terminating training.")
                        logger.mark_early_stop(current_step)
                        process.terminate()
                        break
            except (ValueError, IndexError):
                pass

    process.wait()
    print(f"\n✅ Training complete. Losses saved to {LOSS_LOG_FILE}")

#Run main
if __name__ == "__main__":
    write_config()
    train()