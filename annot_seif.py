import cv2
import os
import torch
from src.utils.helpers import read_json, save_json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device: ",device)
chemin_model="results/weights/finetuned_diwan_augmented_20k_dirichlet_blur_0.3.pth"
model = torch.load(chemin_model, map_location=device)
model.eval()


def predict(img, image_size=(224, 224)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, _, _ = img.shape
    top_cut = height // 6
    bottom_cut = height // 2
    img = img[top_cut : -bottom_cut, :, :]
    img = cv2.resize(img, image_size)
    img = img / 255.0
    img = img.astype(np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        number_logits = output.number_logits       # (1, 100)
        uncertainty = output.uncertainty.item()    # (1,) -> float
        pred_class = number_logits.argmax(dim=1).item()
    return pred_class, uncertainty


# 3. Données
diwan_jersey_annotation_splits = read_json("data/Diwan/train/diwan_jersey_annotation_splits.json")
base_path = "E:/diwan_jersey/bboxes"
annotation_file_path = f"data/diwan/train/seif_train_gt.json"
annotator = 'seif'

if os.path.exists(annotation_file_path):
    annotations = read_json(annotation_file_path)
else:
    annotations = []

crops_annoteted = [crp['relative_path'] for crp in annotations]
print(len(annotations))
print(len([el for el in annotations if el['class'] not  in [-2,-1] ]))

# 4. Boucle principale

cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Annotation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for ctr, game in enumerate(diwan_jersey_annotation_splits[annotator]):
    if game["game"] != "66434b82213237ed872a0865":
        crops = os.listdir(f"{base_path}/{game['game']}/crops/person")
        for crop_ctr, crop in enumerate(crops):
            relative_path = f"{game['game']}/crops/person/{crop}"
            if relative_path in crops_annoteted:
                print("annotated_already: ", relative_path)
                continue

            full_path = f"{base_path}/{relative_path}"
            img = cv2.imread(full_path)
            img_resized = cv2.resize(img, (128, 256))

            # Faire une prédiction avec le modèle
            predicted_class, uncertainty = predict(img)

            # Afficher la prédiction sur l'image
            img_display = img_resized.copy()
            h, w, _ = img_display.shape

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # plus petit
            thickness = 1
            text1 = f"Model: {predicted_class}"
            text2 = f"unc: {uncertainty:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            color = (255, 0, 0)

            # Coordonnées de départ
            x, y = 5, h - 20

            # Dessiner le texte ligne 1
            cv2.putText(img_display, text1, (x, y), font, font_scale, color, thickness)

            # Ligne suivante (avec un petit espacement)
            cv2.putText(img_display, text2, (x, y + 15), font, font_scale, color, thickness)

            # cv2.imshow(f"crop: {crop_ctr}/{game['crops']} | game: {ctr+1}/{len(diwan_jersey_annotation_splits[annotator])} ", img_display)
            cv2.imshow("Annotation" , img_display)

            while True:
                key = cv2.waitKeyEx(0)

                if key == 27:  # ESC
                    print("Exiting.")
                    cv2.destroyAllWindows()
                    exit()

                elif key == ord('x'):  # No number
                    annotations.append({'relative_path': relative_path, 'class': -1})
                    print(f"crop: {crop_ctr}/{game['crops']} || {relative_path.split('/')[-1]} || marked as -1 (No number)")

                elif key == ord('w'):  # Unknown number
                    annotations.append({'relative_path': relative_path, 'class': -2})
                    print(f"crop: {crop_ctr}/{game['crops']} || {relative_path.split('/')[-1]} || marked as -2 (Unknown number)")

                elif key == 13 :  # Confirmer la prédiction du modèle
                    annotations.append({'relative_path': relative_path, 'class': predicted_class})
                    print(f"crop: {crop_ctr}/{game['crops']} || {relative_path.split('/')[-1]} || accepted model prediction: {predicted_class}")

                elif 48 <= key <= 57:  # Taper manuellement un nombre
                    number = chr(key)
                    cv2.putText(img_display, f"Started typing: {number}", (5, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                    cv2.imshow('Image', img_display)
                    key2 = cv2.waitKeyEx(0)
                    if 48 <= key2 <= 57:
                        number += chr(key2)
                    annotations.append({'relative_path': relative_path, 'class': int(number)})
                    print(f"crop: {crop_ctr}/{game['crops']} || {relative_path.split('/')[-1]} || manually entered: {number}")

                else:
                    print("Invalid key. Use 0-9 for numbers, 'w' (no number), 'w' (unknown), 'C' (accept model), ESC to quit.")

                save_json(annotations, annotation_file_path)
                break
            cv2.destroyAllWindows()

    print(f"{game['game']}| {game['crops']} | {ctr+1}/{len(diwan_jersey_annotation_splits[annotator])}")
save_json(annotations, annotation_file_path)
