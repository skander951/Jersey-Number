import cv2
import numpy as np
import os
from collections import Counter

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_number_from_frame(frame):
    return None  # Stub pour OCR futur

def get_top_predict(frames_numbers):
    filtered = [n for n in frames_numbers if n != 0]
    return Counter(filtered).most_common(1)[0][0] if filtered else 0

def create_scroll_animation_with_fixed_window(tracklet_videos, output_path, step=224, max_width=5000):
    all_frames = []
    top_predicts = []

    for v_path in tracklet_videos:
        if not os.path.exists(v_path):
            print(f"❌ Video not found: {v_path}")
            continue
        frames = extract_frames_from_video(v_path)
        if not frames:
            print(f"⚠️ Empty video skipped: {v_path}")
            continue
        all_frames.append(frames)

        numbers = [extract_number_from_frame(f) for f in frames]
        top_predicts.append(get_top_predict(numbers))

    if not all_frames:
        raise ValueError("No valid videos found.")

    frame_height, frame_width, _ = all_frames[0][0].shape
    num_tracklets = len(all_frames)
    overlay_height = 100
    canvas_height = frame_height * num_tracklets + overlay_height
    canvas_width = max_width

    min_len = min(len(f) for f in all_frames)
    total_steps = min_len
    max_steps_to_move = 21  # nombre d'étapes avant que le cadre se fige
    max_scroll_pos = max_steps_to_move * step

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25,
                             (canvas_width, canvas_height))


    for t in range(0, total_steps * step, step // 4):  # mouvement plus fluide
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        for i, frames in enumerate(all_frames):
            y_offset = i * frame_height + overlay_height
            current_idx = min(t // step, len(frames) - 1)
            current_frame = frames[current_idx].copy()

            # Afficher toutes les frames déjà vues (traces)
            for k in range(0, t + 1, step):
                idx = k // step
                if idx < len(frames):
                    xk = k
                    if xk + frame_width > max_width:
                        continue
                    canvas[y_offset:y_offset+frame_height, xk:xk+frame_width] = frames[idx]

            # Position du cadre mobile (bloqué après "max_scroll_pos" pas)
            if t >= max_scroll_pos:
                scroll_x = max_scroll_pos
            else:
                scroll_x = t

            # Afficher le cadre mobile avec transparence
            if scroll_x + frame_width <= max_width:
                alpha = 0.65
                overlay = canvas[y_offset:y_offset + frame_height, scroll_x:scroll_x + frame_width].copy()
                blended = cv2.addWeighted(overlay, 1 - alpha, current_frame, alpha, 0)
                canvas[y_offset:y_offset + frame_height, scroll_x:scroll_x + frame_width] = blended
                cv2.rectangle(canvas, (scroll_x, y_offset),
                              (scroll_x + frame_width, y_offset + frame_height),
                              (0, 0, 0), 3)

        writer.write(canvas)

    writer.release()
    print(f"✅ Video generated and saved at: {output_path}")

# === Test avec tes paths ===
tracklet_paths = [
    "video_top1_prediction/tracklet_video/chelsea/track_7.mp4",
    "video_top1_prediction/tracklet_video/chelsea/track_11.mp4",
    "video_top1_prediction/tracklet_video/chelsea/track_8.mp4"
]

create_scroll_animation_with_fixed_window(tracklet_paths, "video_top1_prediction/scroll.mp4")
