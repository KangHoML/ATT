import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)

    return args

def apply_template(input_text):
    with open('template.json', 'r', encoding='utf-8') as f:
        template = json.load(f)

    formatted_text = f"{template['begin_of_text']}"
    formatted_text += f"{template['start_header']}{template['system_header']}{template['end_header']}\n\n"
    formatted_text += f"{template['system_message']}{template['end_of_text']}"
    formatted_text += f"{template['start_header']}{template['user_header']}{template['end_header']}\n\n"
    formatted_text += f"{template['instruction']}\n"
    formatted_text += f"{input_text}{template['end_of_text']}"

    return formatted_text

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    return model, tokenizer

def test(model, tokenizer, input_text):
    formatted_input = apply_template(input_text)
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=100, 
            num_return_sequences=1,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    assistant_response = generated_text.split(tokenizer.eos_token)[-2].strip()
    
    return assistant_response

if __name__ == "__main__":
    global args
    args = get_args()

    key = "실레입니다만 , 살고 계신 긋이 어디입니까?"
    model, tokenizer = load_model_and_tokenizer()

    corrected_text = test(model, tokenizer, key)
    print(f"결과: {corrected_text}")