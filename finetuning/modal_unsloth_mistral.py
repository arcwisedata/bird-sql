# type: ignore
# ruff: noqa: T201
import os

import modal

app_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("software-properties-common", "git", "curl", "sudo", "htop", "nvtop")
    .pip_install("packaging", "wheel")
    .pip_install("torch==2.3.0")
    .run_commands(
        "pip install 'unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git' --no-build-isolation",
    )
    .pip_install(
        "wandb",
        "boto3",
        "litellm",
        "sqlglot",
        "huggingface_hub==0.23.2",
        "hf-transfer==0.1.5",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
    .entrypoint([])
)

app = modal.App(
    "bird-finetune",
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)

# Volumes for pre-trained models and training runs.
pretrained_volume = modal.Volume.from_name("pretrained-vol")
runs_volume = modal.Volume.from_name("runs-vol")


@app.function(
    image=app_image,
    timeout=86400,
    gpu=os.environ.get("GPU_CONFIG", "h100:1"),
    volumes={
        "/pretrained": pretrained_volume,
        "/runs": runs_volume,
    },
    _allow_background_volume_commits=True,
)
def main(
    train_path: str,
    eval_path: str,
    run_name: str,
):
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
    from unsloth import FastLanguageModel

    max_seq_length = 20_000
    train_data = load_dataset("json", data_files=train_path, split="train")
    eval_data = load_dataset("json", data_files=eval_path, split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/Mistral-Nemo-Instruct-2407",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    def formatting_prompts_func(examples):
        return {
            "text": tokenizer.apply_chat_template(
                examples["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_data = train_data.map(formatting_prompts_func)
    eval_data = eval_data.map(formatting_prompts_func)

    # Only train & evaluate on assistant outputs
    collator = DataCollatorForCompletionOnlyLM(
        response_template="[/INST]",
        tokenizer=tokenizer,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=256,
        lora_dropout=0.0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=42,
        max_seq_length=max_seq_length,
    )

    import wandb

    wandb.init(project="bird-sql", entity="hansonw", name=run_name)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=TrainingArguments(
            # hyperparameters
            num_train_epochs=1.0,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            group_by_length=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            optim="adamw_8bit",
            seed=42,
            fp16=False,
            bf16=True,
            # evaluation
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            bf16_full_eval=True,
            eval_steps=0.2,
            eval_on_start=False,
            # saving/logging
            logging_steps=1,
            output_dir="/runs/" + run_name,
            save_strategy="steps",
            save_steps=0.25,
        ),
    )

    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained("/runs/" + run_name + "/final")
    runs_volume.commit()

    trainer.evaluate()
