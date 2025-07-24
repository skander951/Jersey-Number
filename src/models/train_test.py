import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from time import time
from collections import defaultdict

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(12, 5))
    # Loss subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()
    # Accuracy subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", marker='o')
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", marker='o')
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    plt.show
    return fig

def train(model, train_loader, val_loader, optimizer, loss_fn, device="cuda", num_epochs=50):
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    step = 0

    best_val_loss = float("inf")
    best_weights_by_loss = None

    for epoch in range(num_epochs):
        start_time = time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch["x"].to(device)
            labels = batch["y"].to(device)
            has_pred = batch["has_prediction"].to(device)

            optimizer.zero_grad()
            output = model(images)
            loss, *_ = loss_fn(output, labels, has_pred, step)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            preds = output.number_logits.argmax(dim=1)
            mask = has_pred.bool()
            train_correct += (preds[mask] == labels[mask]).sum().item()
            train_total += mask.sum().item()
            step += 1

        mean_train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["x"].to(device)
                labels = batch["y"].to(device)
                has_pred = batch["has_prediction"].to(device)

                output = model(images)
                val_loss, *_ = loss_fn(output, labels, has_pred, step)
                val_losses.append(val_loss.item())

                preds = output.number_logits.argmax(dim=1)
                mask = has_pred.bool()
                val_correct += (preds[mask] == labels[mask]).sum().item()
                val_total += mask.sum().item()

        mean_val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(val_acc)


        print(f"Train loss: {mean_train_loss:.4f} | Train acc: {train_acc:.4f} | "
              f"Val loss: {mean_val_loss:.4f} | Val acc: {val_acc:.4f} | Time: {time() - start_time:.2f}s ")

        # --- Meilleur modèle basé sur la plus faible val_loss
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_weights_by_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_weights_by_loss)
    return history


def test(model, test_loader, device="cuda", uncertainty_threshold=0.5):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    has_pred_labels = []
    has_pred_model = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["x"].to(device)
            labels = batch["y"].cpu().tolist()
            true_has_pred = batch["has_prediction"].cpu().tolist()

            output = model(images)
            preds = output.number_logits.argmax(dim=1).cpu().tolist()
            uncertainties = output.uncertainty.cpu().tolist()

            for pred, label, true_has, u in zip(preds, labels, true_has_pred, uncertainties):
                if u < uncertainty_threshold :
                    model_has_pred = 1
                else:
                    model_has_pred = 0
                    pred = 0
 
                if true_has_pred == 0:
                    label = 0

                all_preds.append(pred)
                all_labels.append(label)
                has_pred_labels.append(true_has)
                has_pred_model.append(model_has_pred)
    # Évalue la cohérence du masque
    has_pred_labels = np.array(has_pred_labels)
    has_pred_model = np.array(has_pred_model)
    mask_accuracy = (has_pred_labels == has_pred_model).mean()
    print(f"Mask accuracy(unc_threshold={uncertainty_threshold}) : {mask_accuracy:.4f}\n")
    # Métriques classiques
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"Accuracy:  {acc:.4f}\n")
    print(f"Precision: {prec:.4f}\n")
    print(f"Recall:    {rec:.4f}\n")
    print(f"F1-score:  {f1:.4f}\n")

    return {"mask_accuracy" : mask_accuracy,"acc": acc,"precision": prec,"recall": rec,"f1": f1}

def per_digit_accuracy(model, test_loader, device="cuda"):
    model.to(device)
    model.eval()
    correct_by_digit = defaultdict(int)
    total_by_digit = defaultdict(int)
    with torch.no_grad():
        for batch in test_loader:
            images = batch["x"].to(device)
            labels = batch["y"].cpu().tolist()

            output = model(images)
            preds = output.number_logits.argmax(dim=1).cpu().tolist()
  
            for l, p in zip(labels, preds):
                if l != -1 :
                    label = str(l)
                    pred = str(p)
                    
                    for d in label: total_by_digit[d] += 1
                    if len(label) == 1 :

                        if len(pred) == 1:
                            if label == pred :
                                correct_by_digit[label] += 1

                        if len(pred)==2 :
                            for d in pred :
                                if d  == label :
                                    correct_by_digit[label] += 0.5

                                    break
                    elif len(label) == 2 :
                        if len(pred) == 1:
                            for d in label :
                                if d  == pred :
                                    correct_by_digit[d] += 0.5
                                    break
                        elif len(pred) == 2:
                            if (pred[0] == label[0]) and (pred[1] == label[1]) :
                                correct_by_digit[label[0]] += 1
                                correct_by_digit[label[1]] += 1
                            elif (pred[0] == label[0]) :
                                correct_by_digit[label[0]] += 1
                            elif (pred[1] == label[1]) :
                                correct_by_digit[label[1]] += 1
                            elif (pred[0] == label[1]) and (pred[1] == label[0]) :
                                correct_by_digit[label[0]] += 0.5
                                correct_by_digit[label[1]] += 0.5
                            elif (pred[0] == label[1]) :
                                correct_by_digit[label[0]] += 0.5
                            elif (pred[1] == label[0]) :
                                correct_by_digit[label[1]] += 0.5

    digit_accuracy = {
        digit: correct_by_digit[str(digit)] / total_by_digit[str(digit)]
        if total_by_digit[str(digit)] > 0 else 0.0
        for digit in range(10)
    }

    print("Per-digit accuracy: ")
    for digit in range(10):
        print(f"Digit {digit} : {digit_accuracy[digit]:.4f} , {correct_by_digit[str(digit)]} / {total_by_digit[str(digit)]}")

    return digit_accuracy

def grid(model, test_loader, device="cuda", num_images=100, uncertainty_threshold=0.5):
    model.to(device)
    model.eval()

    images_to_show = []
    preds_to_show = []
    labels_to_show = []
    uncertainty_flags = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["x"].to(device)
            labels = batch["y"].to(device)
            has_pred = batch["has_prediction"].to(device)

            output = model(images)
            raw_preds = output.number_logits.argmax(dim=1)
            number_probs = output.number_probs
            uncertainty = output.uncertainty

            for i in range(images.size(0)):
                images_to_show.append(images[i].cpu())

                unc_i = uncertainty[i].item()

                if unc_i < uncertainty_threshold:
                    pred = raw_preds[i].item()
                    prob = number_probs[i, pred].item()
                else:
                    pred = 100
                    prob = None
  
                preds_to_show.append((pred, prob))

                # Remplacement du label s’il est invisible
                label = labels[i].item()
                if has_pred[i].item() == 0:
                    label = 100
                labels_to_show.append(label)

                uncertainty_flags.append(uncertainty[i].item())

                if len(images_to_show) >= num_images:
                    break

    # --- Affichage en grille ---
    cols = 10
    rows = (num_images + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))
    num_display = min(num_images, len(images_to_show))

    for i in range(num_display):
        img = images_to_show[i]
        pred, prob = preds_to_show[i]
        label = labels_to_show[i]
        unc = uncertainty_flags[i]

        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        if (label == 100) and (pred == 100):
            color = "blue"
            title = f"[Invisible]\nPred: {pred}\nU: {unc:.2f}"
        elif pred == 100:
            color = "pink"
            title = f"True: {label}\nPred: [Invisible]\nU: {unc:.2f}"
        else:
            color = "green" if pred == label else "red"
            title = f"True: {label}\nPred: {pred}\nP: {prob:.2f}"

        ax.text(
            5, 20,
            title,
            fontsize=10,
            color=color,
            bbox=dict(facecolor="black", alpha=0.6, pad=3)
        )

    plt.tight_layout()
    return fig


