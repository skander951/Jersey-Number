import cv2
import torch
import numpy as np
from src.tracking.tracking import PlayersTracker
import json
import os
from collections import Counter
from time import time
from tqdm import tqdm

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

def test_video(video_name,tracklet=False,skip_processed=False) :

    video_path = f"data/videos_diwan/{video_name}.mp4"
    os.makedirs("results_video/output_video", exist_ok=True)
    output_path = f"results_video/output_video/{video_name}_detection.mp4"
    if os.path.exists(output_path) and skip_processed : 
        print(f"Skipping {video_name}, already processed.")
    else:
        chemin_model = "results/finetuned_models/ftd_dirichlet_thresh0.6.pth"
        filtre_model_path = "results/weights/filtre_thresh_0.9_basic_params_16.pth"

        model = torch.load(chemin_model, map_location="cuda")
        filtre_model = torch.load(filtre_model_path, map_location="cuda")
        filtre_model.to("cuda").eval()
        model.to("cuda").eval()

        logs = {
            "batch_logs": [],  
            "average_batch_prep_time": 0.0
        }

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

                logs["batch_logs"].append({
                    "frame_id": i,
                    "batch_size": len(crops),
                    "batch_prep_time": batch_prep_time,
                    "filter_model_inference_time": filter_model_inference_time,
                    "filter_post_processing_time": filter_post_processing_time,
                    "valid_inputs": len(valid_inputs),
                    "number_prediction_time": number_prediction_time
                })

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

                        
                    text = f"ID-{det['track_id']} | {pred_label} ({number_prob:.2f})"
                    cv2.rectangle(detection_image, (int(x1 - 5), int(y1 - 30)), (int(x1 + 110), int(y1)), local_color, -1)
                    cv2.circle(detection_image, track_coords, 4, local_color, -1)
                    cv2.putText(detection_image, text, (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            writer.write(detection_image)
            print(f"{i} / {total_frames}")

        for track_id, entries in tracklets.items():
            numbers = [entry["number"] for entry in entries]
            most_common = Counter(numbers).most_common(3)

            summary[track_id] = {
                "appearances": len(entries),
                "top3": []
            }
            for number, count in most_common:
                probs_for_number = [entry["num_prob"] for entry in entries if entry["number"] == number]
                confs_for_number = [entry["filtre_conf"] for entry in entries if entry["number"] == number]
                avg_prob = np.mean(probs_for_number)
                avg_conf = np.mean(confs_for_number)

                summary[track_id]["top3"].append({
                    "number": number,
                    "count": count,
                    "avg_number_probs" : float(avg_prob),
                    "avg_filtre_confidence": float(avg_conf)
                })



        if logs["batch_logs"]:
            logs["average_batch_prep_time"] = round(np.mean([b["batch_prep_time"] for b in logs["batch_logs"]]), 4)
            logs["average_filter_model_inference_time"] = round(np.mean([b["filter_model_inference_time"] for b in logs["batch_logs"]]), 4)
            logs["average_filter_post_processing_time"] = round(np.mean([b["filter_post_processing_time"] for b in logs["batch_logs"]]), 4)
            number_pred_times = [b["number_prediction_time"] for b in logs["batch_logs"] if b["number_prediction_time"] is not None]
            logs["average_number_prediction_time"] = round(np.mean(number_pred_times), 4) if number_pred_times else None

            times_per_image = []
            for b in logs["batch_logs"]:
                total_time = b["batch_prep_time"] + b["filter_model_inference_time"] + b["filter_post_processing_time"]
                if b["number_prediction_time"] is not None:
                    total_time += b["number_prediction_time"]
                batch_size = b["batch_size"] if b["batch_size"] > 0 else 1
                times_per_image.append(total_time / batch_size)

            logs["average_time_per_image"] = round(np.mean(times_per_image), 6)

        os.makedirs("results_video/tracklet_details", exist_ok=True)
        os.makedirs("results_video/tracklet_summary", exist_ok=True)
        os.makedirs("results_video/perf_logs", exist_ok=True)
        os.makedirs("results_video/tracklet_video", exist_ok=True)

        with open(f"results_video/tracklet_details/{video_name}.json", "w") as f:
            json.dump(tracklets, f, indent=4)
        with open(f"results_video/tracklet_summary/{video_name}.json", "w") as f:
            json.dump(summary, f, indent=4)
        with open(f"results_video/perf_logs/{video_name}.json", "w") as f:
            json.dump(logs, f, indent=4)

        if tracklet:
            tracklet_video_dir = f"results_video/tracklet_video/{video_name}"
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

                        text = f"{number} ({prob:.2f})"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(crop_resized, (5, 5), (5 + text_width + 4, 5 + text_height + 4), (0, 0, 0), -1)
                        cv2.putText(crop_resized, text, (7, 7 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cropped_frames.append(crop_resized)

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

        cap.release()
        writer.release()
        cv2.destroyAllWindows()



video_names = ["asm_ass1","aso_scb1","ass_usbg1","ca_bg1","derby","esm_cab1","est_bg1","est_css1","usmo_ess1","ust_ob1"]
for name in video_names:
    test_video(name,tracklet=True,skip_processed=True)
