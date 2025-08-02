import cv2
import torch
import numpy as np
from src.tracking.tracking import PlayersTracker
import os
from collections import Counter
from time import time
from tqdm import tqdm

base_path = "C:/Users/skouz/OneDrive/Documents/Jersey-Number/"
pl_tr = PlayersTracker(base_path=base_path, track_thresh=0.9,track_buffer=100,match_thresh=0.9,frame_rate=25,minimum_consecutive_frames=1)

def put_bold_text(img, text, org, font, font_scale, color, thickness):
    x, y = org
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (x + dx, y + dy), font, font_scale, color, thickness)
    cv2.putText(img, text, org, font, font_scale, color, thickness)

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

def test_video(video_name,tracklet=False,skip_processed=False) :

    video_path = f"data/videos_diwan/{video_name}.mp4"
    os.makedirs("results_video1/output_video", exist_ok=True)
    output_path = f"results_video1/output_video/{video_name}2.mp4"
    chemin_model = "results/finetuned_models/ftd_dirichlet_thresh0.6.pth"
    filtre_model_path = "results/weights/filtre_thresh_0.9_basic_params_16.pth"

    model = torch.load(chemin_model, map_location="cuda")
    filtre_model = torch.load(filtre_model_path, map_location="cuda")
    filtre_model.to("cuda").eval()
    model.to("cuda").eval()


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    tracklets = {}
    summary = {}

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
            st1 = time()
            batchs = batch_prep(crops)
            batch_prep_time = round(time() - st1, 4)
            print(f"[Batch size: {len(crops)}] Batch prep time: {batch_prep_time}s")

            with torch.no_grad():
                st2 = time()
                pred_filtre = filtre_model(batchs)
                filter_model_inference_time = round(time() - st2, 4)
                print(f"[Batch size: {len(crops)}] Filter model inference time: {filter_model_inference_time}s")
                probs = torch.sigmoid(pred_filtre)             
                pred_filtre_labels = (probs > 0.5).long().squeeze(1).tolist()

                valid_inputs = []
                valid_indices = []

                for idx, val in enumerate(pred_filtre_labels):
                    if val == 1: 
                        valid_inputs.append(batchs[idx])
                        valid_indices.append(idx)
                filter_post_processing_time = round(time() - st2, 4)
                print(f"[Batch size: {len(crops)}] Filter post-processing time: {filter_post_processing_time}s")

                if len(valid_inputs) > 0:
                    st3 = time()
                    valid_tensor = torch.stack(valid_inputs)
                    output = model(valid_tensor)
                    number_prediction_time = round(time() - st3, 4)
                    pred_number = torch.argmax(output.number_logits, dim=1).tolist()
                    print(f"[Valid inputs: {len(valid_inputs)}] Number prediction time: {number_prediction_time}s")
                else:
                    pred_number = []
                    number_prediction_time = None


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
                    number_prob = 0.0
                    local_color = (0, 0, 255) 

                else:
                    j_idx = valid_indices.index(idx)
                    pred_label = pred_number[j_idx] 
                    confidence = output.number_probs[j_idx][pred_label].item()
                    number_prob = jersey_probs[j_idx][pred_label] if jersey_probs is not None else 0.0
                    local_color = (0, 255, 0)

                track_id = det["track_id"]
                if track_id not in tracklets:
                    tracklets[track_id] = []

                tracklets[track_id].append({
                    "frame_id": i,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "number": int(pred_label),
                    "num_prob": float(number_prob),
                    "filtre_conf": float(confidence)
                    
                })

                    
                text = f"ID_{det['track_id']} | {pred_label} ({number_prob:.2f})"
                cv2.rectangle(detection_image, (int(x1 - 5), int(y1 - 30)), (int(x1 + 110), int(y1)), local_color, -1)
                cv2.circle(detection_image, track_coords, 4, local_color, -1)
                cv2.putText(detection_image, text, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        writer.write(detection_image)
        print(f"{i} / {total_frames}")

    for track_id, entries in tracklets.items():
        numbers = [entry["number"] for entry in entries]
        most_common = Counter(numbers).most_common(3)

        top1_number = most_common[0][0] if most_common else 0

        filtered_numbers = [n for n in numbers if n != 0]
        if filtered_numbers:
            top_predict = Counter(filtered_numbers).most_common(1)[0][0]
        else:
            top_predict = 0

        summary[track_id] = {
            "appearances": len(entries),
            "top1_number": int(top1_number),
            "top_predict": int(top_predict)
        }


    os.makedirs("results_video1/tracklet_video", exist_ok=True)

    if tracklet:
        tracklet_video_dir = f"results_video1/tracklet_video/{video_name}"
        if os.path.exists(tracklet_video_dir):
            print(f"Skipping tracklets for {video_name}, already processed.")
        else:
            os.makedirs(tracklet_video_dir, exist_ok=True)

        for track_id, entries in tqdm(tracklets.items(), desc="Generating Tracklet Videos"):
            if len(entries) < 120:
                continue

            video_out_path = os.path.join(tracklet_video_dir, f"track_{track_id}.mp4")
            cropped_frames = []


            for entry in tqdm(entries, desc=f"Track {track_id} Frames", leave=False):
                frame_id = entry["frame_id"]
                bbox = entry["bbox"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    continue
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]

                crop_resized = cv2.resize(crop, (224, 224))
                number = entry["number"]
                prob = entry["num_prob"]

                # Texte à afficher
                text_pred = f"Pred : {number}"
                text_prob = f"Prob : {prob:.2f}"

                # Calculer la largeur/hauteur nécessaires
                (text_width_pred, text_height_pred), _ = cv2.getTextSize(text_pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (text_width_prob, text_height_prob), _ = cv2.getTextSize(text_prob, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Définir la hauteur totale du bandeau (padding + interligne)
                padding = 5
                spacing = 4
                band_height = text_height_pred + text_height_prob + spacing + 2 * padding
                band_width = max(text_width_pred, text_width_prob) + 2 * padding

                # Créer une nouvelle image plus haute (bandeau + image crop)
                total_height = 224 + band_height
                annotated_frame = np.ones((total_height, 224, 3), dtype=np.uint8) * 255  # image blanche

                # Coller l’image crop en bas
                annotated_frame[band_height:, :, :] = crop_resized

                put_bold_text(annotated_frame, text_pred, (padding, padding + text_height_pred), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                put_bold_text(annotated_frame, text_prob, (padding, padding + text_height_pred + spacing + text_height_prob), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                cropped_frames.append(annotated_frame)

            if len(cropped_frames) > 0:
                out_writer = cv2.VideoWriter(
                    video_out_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (224, total_height)  
                )
                for frame_crop in cropped_frames:
                    out_writer.write(frame_crop)
                out_writer.release()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

video_names = ["chelsea"]
for name in video_names:
    test_video(name,tracklet=False,skip_processed=True)
