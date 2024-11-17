from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, MergeModelCallback
from trl.mergekit_utils import MergeConfig
import tempfile
import gc
import os
import shutil
from transformers.trainer_utils import get_last_checkpoint

model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

print("testing")
output_dir = "trained_dir"
training_args = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    report_to="none",
    save_strategy="steps",
    save_steps=1,
)
print("training args created")
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
last_checkpoint = get_last_checkpoint(output_dir)
print("last checkpoint: ", last_checkpoint)
merged_path = os.path.join(last_checkpoint, "merged")