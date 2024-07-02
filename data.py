import os
import json

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def make_dataset(root, train=True):
    dataset = []
    
    # train & validation에 따른 경로
    if train:
        dir_path = os.path.join(root, "Training/")
        data_path = os.path.join(dir_path, "train_data.json")
    else:
        dir_path = os.path.join(root, "Validation/")
        data_path = os.path.join(dir_path, "valid_data.json")

    # 각 json 파일의 데이터를 template에 맞게 변환
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for item in data:
            dataset.append(item)
    
    # dataset으로 변환하여 반환
    return Dataset.from_list(dataset)

if __name__ == "__main__":
    # 데이터셋 구성
    root = "../datasets/KoreanError"
    train_dataset, val_dataset = make_dataset(root, train=True), make_dataset(root, train=False)

    # Dictionary 형태로 변환
    data_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset
    })

    # HuggingFace에 로그인
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    # Hub에 업로드
    data_addr = "kanghokh/att_data"
    data_dict.push_to_hub(data_addr)
    
    
    