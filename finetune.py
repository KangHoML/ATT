import os
import json
import torch
import argparse

from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

parser = argparse.ArgumentParser()
# -- dataset
parser.add_argument("--data", type=str, default="kanghokh/att_data")

# -- model
parser.add_argument("--base_model", type=str, default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
parser.add_argument("--save_path", type=str, default="./results/try2")

# -- lora config
parser.add_argument("--alpha", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--rank", type=int, default=64)

# -- training configuration
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--norm", type=float, default=0.3)
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--lr_scheduler", type=str, default="constant")
parser.add_argument("--steps", type=int, default=1000)

# checkpoint로 훈련 재개
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--log_steps", type=int, default=100)

# bfloat & flash_attention available
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    compute_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    compute_dtype = torch.float16

# apply the pre-defined template
def apply_template(example):
    with open('template.json', 'r') as f:
        template = json.load(f)

    system_part = f"{template['system_header']} {template['system_message']}{template['eot_token']}"
    user_part = f"{template['user_header']} {template['user_instruction']}\n\n{example['err_sentence']}{template['eot_token']}"
    assistant_part = f"{template['assistant_header']} {example['cor_sentence']}{template['eot_token']}"
    
    return {"text": f"{system_part}{user_part}{assistant_part}"}    

# add eos token
def add_eos(example, tokenizer):
    if not example['text'].endswith(tokenizer.eos_token):
        example['text'] += tokenizer.eos_token
    return example

def train():
    # load the dataset
    dataset = load_dataset(args.data)
    train_dataset, val_dataset = dataset['train'], dataset['val']

    # apply the template
    train_dataset = train_dataset.map(apply_template)
    val_dataset = val_dataset.map(apply_template)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    # pad token setting
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # add EOS token if needed
    train_dataset = train_dataset.map(lambda x: add_eos(x, tokenizer))
    val_dataset = val_dataset.map(lambda x: add_eos(x, tokenizer))

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation
    )
    model.config.use_cache = False


    # quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False
    )

    # peft configuration
    lora_config = LoraConfig(
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        r=args.rank,
        bias="none",
        task_type="CASUAL_LM"
    )

    # training configuration
    train_args = TrainingArguments(
        # basis
        output_dir=args.save_path,
        num_train_epochs=args.epoch,

        # data loader
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        group_by_length=True,

        # optimizer setting
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.norm,
        optim="paged_adamw_32bit",

        # learnning rate scheduler
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup,

        # gradient accumulation
        gradient_accumulation_steps=4,

        # precision
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),

        # evaluation
        evaluation_strategy="steps",
        eval_steps=args.steps,
        save_strategy="steps",
        save_steps=args.steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # log
        logging_dir=f"{args.save_path}/logs",
        logging_steps=args.log_steps,
        report_to="tensorboard",
        disable_tqdm=False
    )

    # train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=train_args,
        packing=False
    )

    if args.ckpt_path is not None:
        trainer.train(resume_from_checkpoint=args.ckpt_path)
    else:
        trainer.train()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    # train 및 결과 출력
    os.makedirs(args.save_path, exist_ok=True)
    train()