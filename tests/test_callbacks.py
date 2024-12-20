# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import tempfile
import unittest

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments
from transformers.testing_utils import require_peft, require_wandb
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_peft_available

from tests.testing_utils import require_mergekit
from trl import BasePairwiseJudge, DPOConfig, DPOTrainer, LogCompletionsCallback, MergeModelCallback, WinRateCallback
from trl.import_utils import is_mergekit_available


if is_peft_available():
    from peft import LoraConfig

if is_mergekit_available():
    from trl.mergekit_utils import MergeConfig


class HalfPairwiseJudge(BasePairwiseJudge):
    """Naive pairwise judge that always returns [1, 0]"""

    def judge(self, prompts, completions, shuffle_order=True):
        # just check that the batch size is 2
        assert len(prompts) == 2
        return [1, 0]


class TrainerWithRefModel(Trainer):
    # This is a dummy class to test the callback. Compared to the Trainer class, it only has an additional
    # ref_model attribute
    def __init__(self, model, ref_model, args, train_dataset, eval_dataset, processing_class):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
        )
        self.ref_model = ref_model


class WinRateCallbackTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        dataset["train"] = dataset["train"].select(range(8))
        self.expected_winrates = [
            {"eval_win_rate": 0.5, "epoch": 0.0, "step": 0},
            {"eval_win_rate": 0.5, "epoch": 0.5, "step": 2},
            {"eval_win_rate": 0.5, "epoch": 1.0, "step": 4},
            {"eval_win_rate": 0.5, "epoch": 1.5, "step": 6},
            {"eval_win_rate": 0.5, "epoch": 2.0, "step": 8},
            {"eval_win_rate": 0.5, "epoch": 2.5, "step": 10},
            {"eval_win_rate": 0.5, "epoch": 3.0, "step": 12},
        ]

        def tokenize_function(examples):
            out = self.tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(tokenize_function, batched=True)

        self.generation_config = GenerationConfig(max_length=32)
        self.judge = HalfPairwiseJudge()

    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = TrainerWithRefModel(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)

    def test_without_ref_model(self):
        # Same as before, but without the ref_model attribute. It should use the model attribute instead
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)

    @require_peft
    def test_lora(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.add_adapter(peft_config)
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)


@require_wandb
class LogCompletionsCallbackTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        dataset["train"] = dataset["train"].select(range(8))

        def tokenize_function(examples):
            out = self.tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(tokenize_function, batched=True)

        self.generation_config = GenerationConfig(max_length=32)

    def test_basic(self):
        import wandb

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="wandb",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            completions_callback = LogCompletionsCallback(trainer, self.generation_config, num_prompts=2)
            trainer.add_callback(completions_callback)
            trainer.train()

            # Get the current run
            completions_path = wandb.run.summary.completions["path"]
            json_path = os.path.join(wandb.run.dir, completions_path)
            with open(json_path) as f:
                completions = json.load(f)

            # Check that the columns are correct
            self.assertIn("step", completions["columns"])
            self.assertIn("prompt", completions["columns"])
            self.assertIn("completion", completions["columns"])

            # Check that the prompt is in the log
            self.assertIn(self.dataset["test"][0]["prompt"], completions["data"][0])


@require_mergekit
class MergeModelCallbackTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    def test_last_checkpoint(self):
        output_dir = "dir_last_checkpoint"
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
        )
        config = MergeConfig("linear")
        merge_callback = MergeModelCallback(config, push_to_hub=False, merge_at_every_checkpoint=False)
        trainer.add_callback(merge_callback)
        trainer.train()

        last_checkpoint = get_last_checkpoint(output_dir)
        merged_path = os.path.join(last_checkpoint, "merged")
        self.assertTrue(os.path.isdir(merged_path), "Merged folder does not exist in the last checkpoint.")

    def test_every_checkpoint(self):
        output_dir = "dir_every_checkpoint"
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
        )
        config = MergeConfig("linear")
        merge_callback = MergeModelCallback(config, push_to_hub=False, merge_at_every_checkpoint=True)
        trainer.add_callback(merge_callback)
        trainer.train()

        checkpoints = sorted(
            [os.path.join(output_dir, cp) for cp in os.listdir(output_dir) if cp.startswith("checkpoint-")]
        )

        for checkpoint in checkpoints:
            merged_path = os.path.join(checkpoint, "merged")
            self.assertTrue(os.path.isdir(merged_path), f"Merged folder does not exist in checkpoint {checkpoint}.")
