import cv2
import torch
import numpy as np
from src.tracking.tracking import PlayersTracker
import numpy as np
import json
import os
from collections import Counter
from time import time

base_path = "C:/Users/skouz/OneDrive/Documents/GitHub/jersey_id/"
pl_tr = PlayersTracker(base_path=base_path, track_thresh=0.9,track_buffer=100,match_thresh=0.9,frame_rate=25,minimum_consecutive_frames=1)

def jersey_crop_image_prep(frame,bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
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
    
video_name = "est_bg1"
video_path = f"data/videos for test/{video_name}.mp4"
output_path = f"results_tracklet/videos/{video_name}_detection.mp4"
chemin_model = "results/weights/finetuned_diwan_augmented_20k_dirichlet_blur_0.3.pth"
filtre_model_path = "results/weights/filtre_thresh_0.9_basic_params_16.pth"
tracklet_video_dir = f"results_tracklet/tracklet_videos/{video_name}"
os.makedirs(tracklet_video_dir, exist_ok=True)

st = time()
st1 = time()

model = torch.load(chemin_model, map_location="cuda")
filtre_model = torch.load(filtre_model_path, map_location="cuda")
filtre_model.to("cuda").eval()
model.to("cuda").eval()

print(f"time to load models{round(time()-st1,4)}")

# read video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
tracklets = {}
summary = {}

st2 = time()
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        continue

    detection_image = np.copy(frame)
    detections, tracker_ids = pl_tr.track_with_yolo(detection_image, persist=True, conf=0.25, iou=0.35)
    frame_outputs = []
    crops = []

    if len(detections) > 0:
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2, h, w = int(det[0]), int(det[1]), int(det[2]), int(det[3]), int(det[4]), int(det[5])
            track_id = tracker_ids[det_idx]
            track_coords = (int((x1 + x2 / 2)), int(y1 + y2))
            crop = jersey_crop_image_prep(frame, (x1, y1, x2, y2))
            crops.append(crop)

            frame_output = {
                "track_id": int(track_id),
                "video_coords": track_coords,
                "bbox_xyxy": (x1, y1, x2, y2)
            }
            frame_outputs.append(frame_output)

        batchs = batch_prep(crops)

        with torch.no_grad():
            # Prédiction avec le filtre
            pred_filtre = filtre_model(batchs) 
            probs = torch.sigmoid(pred_filtre)             
            pred_filtre_labels = (probs > 0.5).long().squeeze(1).tolist()

            # Liste pour les crops valides à passer au modèle de jersey
            valid_inputs = []
            valid_indices = []

            for idx, val in enumerate(pred_filtre_labels):
                if val == 1: 
                    valid_inputs.append(batchs[idx])
                    valid_indices.append(idx)

            if len(valid_inputs) > 0:
                valid_tensor = torch.stack(valid_inputs)
                output = model(valid_tensor)
                pred_number = torch.argmax(output.number_logits, dim=1).tolist()
            else:
                pred_number = []

        pred_filtre_probs = probs.cpu().numpy()

        if len(valid_inputs) > 0:
            jersey_probs = output.number_probs.cpu().numpy()
        else:
            jersey_probs = []

        for idx, det in enumerate(frame_outputs):
            x1, y1, x2, y2 = det["bbox_xyxy"]
            track_coords = det["video_coords"]

            if pred_filtre_labels[idx] == 0:
                pred_label = 0
                confidence = pred_filtre_probs[idx][0]
                local_color = (0, 0, 255) 

            else:
                j_idx = valid_indices.index(idx)
                pred_label = pred_number[j_idx] 
                confidence = output.number_probs[j_idx][pred_label].item()
                local_color = (0, 255, 0)

            track_id = det["track_id"]
            if track_id not in tracklets:
                tracklets[track_id] = []

            tracklets[track_id].append({
                "frame_id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "number": int(pred_label),
                "conf": float(confidence)
            })

                
            text = f"ID-{det['track_id']} | {pred_label} ({confidence:.2f})"
            cv2.rectangle(detection_image, (int(x1 - 5), int(y1 - 30)), (int(x1 + 110), int(y1)), local_color, -1)
            cv2.circle(detection_image, track_coords, 4, local_color, -1)
            cv2.putText(detection_image, text, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    writer.write(detection_image)
    print(f"{i} / {total_frames}")

print(f"time inference {round(time()-st2,4)}")
st3 = time()

for track_id, entries in tracklets.items():
    numbers = [entry["number"] for entry in entries]
    confs = [entry["conf"] for entry in entries]
    most_common = Counter(numbers).most_common(3)

    summary[track_id] = {
        "appearances": len(entries),
        "top3": []
    }
    for number, count in most_common:
        confs_for_number = [entry["conf"] for entry in entries if entry["number"] == number]
        avg_conf = np.mean(confs_for_number)
        summary[track_id]["top3"].append({
            "number": number,
            "count": count,
            "avg_confidence": float(avg_conf)
        })

with open(f"results_tracklet/videos_json/{video_name}_full_tracklets.json", "w") as f:
    json.dump(tracklets, f, indent=4)
with open(f"results_tracklet/videos_json/{video_name}_tracklet_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print(f"time to save jsons {round(time()-st3,4)}")
st4 = time()

cap = cv2.VideoCapture(video_path)

for track_id, entries in tracklets.items():
    if len(entries) < 25:
        continue 

    # Définir le nom du fichier de sortie pour ce joueur
    video_out_path = os.path.join(tracklet_video_dir, f"track_{track_id}.mp4")

    # Liste pour sauvegarder les frames recadrées
    cropped_frames = []

    for entry in entries:
        frame_id = entry["frame_id"]
        bbox = entry["bbox"]

        # Aller à la bonne frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        # Redimensionner pour garder une taille uniforme
        crop_resized = cv2.resize(crop, (224, 224))
        number = entry["number"]
        conf = entry["conf"]
        text = f"{number} ({conf:.2f})"
        cv2.putText(crop_resized, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cropped_frames.append(crop_resized)

    # Création de la vidéo pour ce track_id
    if len(cropped_frames) > 0:
        out_writer = cv2.VideoWriter(
            video_out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (224, 224)
        )
        for frame_crop in cropped_frames:
            out_writer.write(frame_crop)
        out_writer.release()

print(f"time to save tracklets videos {round(time()-st4,4)}")

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"total time {round(time()-st,4)}")


