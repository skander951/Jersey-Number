import cv2
import os
import torch
from src.utils.helpers import read_json, save_json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chemin_model="results/finetuned_models/ftd_dirichlet_thresh0.6.pth"
model = torch.load(chemin_model, map_location=device)
model.eval()


def predict(img, image_size=(224, 224)):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, _, _ = img.shape
    top_cut = height // 6
    bottom_cut = height // 2
    img = img[top_cut : -bottom_cut, :, :]

    # 3. Resize
    img = cv2.resize(img, image_size)

    # 4. Normalize to [0, 1]
    img = img / 255.0
    img = img.astype(np.float32)

    # 5. Convert to tensor (C, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # 6. Predict
    with torch.no_grad():
        output = model(tensor)
        number_logits = output.number_logits      
        number_probs = output.number_probs         

        pred_class = number_logits.argmax(dim=1).item()   
        prob_num = number_probs[0, pred_class].item() 
        

    return pred_class, prob_num


# 3. Données
diwan_jersey_annotation_splits = read_json("data\\diwan\\train\\diwan_jersey_annotation_splits.json")
base_path = "data\\diwan\\train\\images"
annotation_file_path = f"data\\diwan\\train\\train_gt.json"
annotator = 'skander'

if os.path.exists(annotation_file_path):
    annotations = read_json(annotation_file_path)
else:
    annotations = []

crops_annoteted = [crp['relative_path'] for crp in annotations]

# 4. Boucle principale
for ctr, game in enumerate(diwan_jersey_annotation_splits[annotator]):
    if game["game"] != "66434b82213237ed872a0865":
        crops = os.listdir(f"{base_path}/{game['game']}/crops/person")
        for crop in crops:
            relative_path = f"{game['game']}/crops/person/{crop}"
            if relative_path in crops_annoteted:
                print("annotated_already: ", relative_path)
                continue

            full_path = f"{base_path}/{relative_path}"
            img = cv2.imread(full_path)
            img_resized = cv2.resize(img, (128, 256))

            # Faire une prédiction avec le modèle
            predicted_class, prob_num = predict(img)

            # Afficher la prédiction sur l'image
            img_display = img_resized.copy()
            h, w, _ = img_display.shape

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # plus petit
            thickness = 1
            text1 = f"Model: {predicted_class}"
            text2 = f"prob: {prob_num:.2f}"

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

            cv2.imshow('Image', img_display)

            while True:
                key = cv2.waitKeyEx(0)

                if key == 27:  # ESC
                    print("Exiting.")
                    cv2.destroyAllWindows()
                    exit()

                elif key == ord('X'):  # No number
                    annotations.append({'relative_path': relative_path, 'class': -1})
                    print(f"{relative_path} || marked as -1 (No number)")

                elif key == ord('W'):  # Unknown number
                    annotations.append({'relative_path': relative_path, 'class': -2})
                    print(f"{relative_path} || marked as -2 (Unknown number)")

                elif key == ord('C'):  # Confirmer la prédiction du modèle
                    annotations.append({'relative_path': relative_path, 'class': predicted_class})
                    print(f"{relative_path} || accepted model prediction: {predicted_class}")

                elif 48 <= key <= 57:  # Taper manuellement un nombre
                    number = chr(key)
                    cv2.putText(img_display, f"Started typing: {number}", (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Image', img_display)
                    key2 = cv2.waitKeyEx(0)
                    if 48 <= key2 <= 57:
                        number += chr(key2)
                    annotations.append({'relative_path': relative_path, 'class': int(number)})
                    print(f"{relative_path} || manually entered: {number}")

                else:
                    print("Invalid key. Use 0-9 for numbers, 'X' (no number), 'W' (unknown), 'C' (accept model), ESC to quit.")

                save_json(annotations, annotation_file_path)
                break

    print(f"{game} | {ctr+1}/{len(diwan_jersey_annotation_splits[annotator])}")
