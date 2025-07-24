import json
import os
import torch

def save_json(data, path):
    jsonString = json.dumps(data)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    print("saved")

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def chunkify(big_list, chunk_size):
    chunks = [big_list[x:x + chunk_size] for x in range(0, len(big_list), chunk_size)]
    return chunks

def count_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print("="*50)
    print(f"📊 Total Parameters in Model: {total:,}".center(50))
    print("="*50)



