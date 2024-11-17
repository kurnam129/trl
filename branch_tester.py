from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, MergeModelCallback
from trl.mergekit_utils import MergeConfig
import tempfile
import gc
import os

def set_permissions_for_safetensors(folder_path, mode=0o777):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                os.chmod(file_path, mode)  # Change permissions
                print(f"Permissions set to {oct(mode)} for: {file_path}")


model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = DPOConfig(
        output_dir=tmp_dir,
        num_train_epochs=1,
        report_to="none",
        save_strategy="steps",
        save_steps=1,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer)

    print("trainer created")
    config = MergeConfig("linear")
    print("config created")
    merge_callback = MergeModelCallback(config, push_to_hub=False, merge_at_every_checkpoint=False)
    print("callback created")
    trainer.add_callback(merge_callback)
    print("callback added")
    trainer.train()
    print("training done")
    set_permissions_for_safetensors(tmp_dir)
    print("permission changed")
del trainer
print("trainer deleted")
gc.collect()
print("garbage collected")
