import cv2
import os
import torch
from src.utils.helpers import read_json, save_json, chunkify
import numpy as np
from time import time


def jersey_crop_image_prep(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    height_crop, width_crop, _ = crop_rgb.shape
    top_cut = height_crop // 6
    bottom_cut = height_crop // 2
    crop_rgb = crop_rgb[top_cut : height_crop - bottom_cut, :, :]
    crop_rgb = cv2.resize(crop_rgb, (224, 224))
    return crop_rgb

def batch_prep(image_list):
    image_list = np.array(image_list)
    image_list = image_list / 255.0
    image_list = image_list.astype(np.float32)
    input_tensor = torch.from_numpy(image_list).permute(0,3, 1, 2).to("cuda")
    return input_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)

st1 = time()
legibility_model_path = "results/weights/filtre_thresh_0.9_basic_params_16.pth"
legibility_model = torch.load(legibility_model_path, map_location="cuda")
legibility_model.eval()
print(f"time to load models: {round(time()-st1,4)}")


crops_path = "E:/diwan_jersey/v2/bboxes"
annotation_legib_file_path = f"data/diwan_v2/diwan_v2_with_legib.json"

diwan_v2_with_legib = []

batch_size=32
games = os.listdir(crops_path)

st2 = time()
legib_crops = 0
for ct_g, game in enumerate(games):
    st_game = time()
    game_crops_list = os.listdir(f"{crops_path}/{game}/crops/person")
    chunked_game_crops_list = chunkify(game_crops_list, batch_size)
    for chunk in chunked_game_crops_list:
        clean_crops_list = []
        for crop_name in chunk:
            crop = cv2.imread(f"{crops_path}/{game}/crops/person/{crop_name}")
            clean_crop  = jersey_crop_image_prep(crop)
            clean_crops_list.append(clean_crop)
        crops_batch = batch_prep(clean_crops_list)
        with torch.no_grad():
            pred_filtre = legibility_model(crops_batch) 
            probs = torch.sigmoid(pred_filtre)             
            pred_filtre_labels = (probs > 0.5).long().squeeze(1).tolist()
            probs_list = [round(p.item(), 6) for p in probs.squeeze(1)]
            # print(pred_filtre_labels)
        for crp,pb in zip(chunk, probs_list):
            relative_path = f"{game}/crops/person/{crp}"
            diwan_v2_with_legib.append({'relative_path': relative_path, 'legib_prob': pb})
        legib_crops+= pred_filtre_labels.count(1)
    print(f"{ct_g+1}/{len(games)} | {game} | crops: {len(game_crops_list)} | batches_of_{batch_size}: {len(chunked_game_crops_list)} | took: {round(time()-st_game,4)} s |  | ")
    print("="*120)


print(f"Processing games took: {round(time()-st2,4)} s")

print('legib crops: ', legib_crops)

save_json(diwan_v2_with_legib, annotation_legib_file_path)




