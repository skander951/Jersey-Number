import os
import cv2
import os
import cv2
import random
import tensorflow as tf
import numpy as np
from time import time
from ultralytics import YOLO


games_path = "/mnt/e/games/22-23/Championnat de Tunisie"
games_path = "/mnt/e/highlights/22-23"

games = os.listdir(games_path)



seg_img_size = (960//2, 560 // 2)

random.seed(42)

ouput_path = "/mnt/e/diwan_jersey/v2/frames"

yolo_model = YOLO("yolov10n.pt", task='detect')

segmentation_model_path = f"models/alexnet_segmentation_loss_bc_epochs20_seed16_nodes128_run_0_loss"
segmentation_model = tf.keras.models.load_model(segmentation_model_path)
# segmentation_model  = tf.keras.layers.TFSMLayer("models/alexnet_segmentation_loss_bc_epochs20_seed16_nodes128_run_0_loss", call_endpoint='serving_default')

main_smaple, max_frames = 100, 12


games_processed_already = os.listdir("/mnt/e/diwan_jersey/v2/bboxes")
for g_ct, game in enumerate(games):
    videos = os.listdir(f"{games_path}/{game}")
    if game in games_processed_already:
        print(g_ct, game, len(videos), "PROCESSED ALREADY !!!")
        continue
    print(game, len(videos))
    if len(videos)>0:
        try: os.mkdir(f"{ouput_path}/{game}")
        except: pass

        st = time()
        video_path = f"{games_path}/{game}/{videos[0]}"
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_samle = random.sample([i for i in range(total_frames)], main_smaple)
        frames_samle = sorted(frames_samle)
        frames_list = []
        seg_frames_list = []
        for frame_idx in frames_samle:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            if ret:
                frames_list.append(frame)
                seg_frame = cv2.resize(frame, seg_img_size)
                seg_frames_list.append(seg_frame)
        seg_frames_array = np.array(seg_frames_list).astype('float32') / 255.0
        seg_preds = segmentation_model.predict(seg_frames_array,verbose=0)
        predictions = [float(prd[0]) for prd in seg_preds]
        yolo_frames = []
        for idx, pred in enumerate(predictions):
            if pred < 0.2:
                yolo_frames.append(frames_list[idx])
                cv2.imwrite(f"{ouput_path}/{game}/frame_{idx}.jpg", frames_list[idx])
        conf = 0.35
        if len(yolo_frames)>max_frames:
            yolo_model.predict(yolo_frames[:max_frames], conf=conf, classes=[0], project = "/mnt/e/diwan_jersey/v2/bboxes", name=game,verbose =True, save_crop=True) 
        else:
            yolo_model.predict(yolo_frames, conf=conf, classes=[0], project = "/mnt/e/diwan_jersey/v2/bboxes", name=game,verbose =False, save_crop=True) 
        print(g_ct+1, len(games),game, len(yolo_frames), f"took: {round(time()-st,2)}")
   