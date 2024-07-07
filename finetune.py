import os
import json
import torch
import argparse

from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    
    # configuration 파일 열기
    with open(args.config, 'r') as f:
        config = json.load(f)

    # argparser Namespace 객체로 변환
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)

    return args

# bfloat & flash_attention available
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    compute_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    compute_dtype = torch.float16

# LLM Template 
def apply_template(example):
    with open('template.json', 'r', encoding='utf-8') as f:
        template = json.load(f)

    formatted_text = f"{template['begin_of_text']}"
    
    # System message
    formatted_text += f"{template['start_header']}{template['system_header']}{template['end_header']}\n\n"
    formatted_text += f"{template['system_message']}{template['end_of_text']}"
    
    # User message (instruction and input)
    formatted_text += f"{template['start_header']}{template['user_header']}{template['end_header']}\n\n"
    formatted_text += f"{template['instruction']}\n"
    formatted_text += f"{example['err_sentence']}{template['end_of_text']}"
    
    # Assistant message (corrected sentence)
    formatted_text += f"{template['start_header']}{template['assistant_header']}{template['end_header']}\n\n"
    formatted_text += f"{example['cor_sentence']}{template['end_of_text']}"

    return {"text": formatted_text}

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
        token=True,
        trust_remote_code=True
    )

    # pad token setting
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # add EOS token if needed
    train_dataset = train_dataset.map(lambda x: add_eos(x, tokenizer))
    val_dataset = val_dataset.map(lambda x: add_eos(x, tokenizer))

    # quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False
    )
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation
    )
    model.config.use_cache = False

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
        lr_scheduler_type=args.type,
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
        metric_for_best_model="eval_samples_per_second",
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
        trainer.train(resume_from_checkpoint=f"{args.save_path}/{args.ckpt_path}")
    else:
        trainer.train()

if __name__ == "__main__":
    global args
    args = get_args()

    # train 및 결과 출력
    os.makedirs(args.save_path, exist_ok=True)
    train()