import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from time import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score,\
                        f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import LambdaLR


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, basic_params=64):
        super(ResNetBinaryClassifier, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.conv1 = nn.Conv2d(3, basic_params, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(basic_params)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(basic_params, basic_params * 2, stride=2)
        self.layer2 = BasicBlock(basic_params * 2, basic_params * 4, stride=2)
        self.layer3 = BasicBlock(basic_params * 4, basic_params * 8, stride=2)

        self.attention = nn.Sequential(
            nn.Conv2d(basic_params * 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(basic_params * 8, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        attention_mask = self.attention(x)
        x = x * attention_mask
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


    def train_legib(self, num_epochs=50, train_loader=None, valid_loader=None, device='cpu', lr=1e-4):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = self.criterion

        def lr_lambda(epoch):
            if epoch < 20:
                return 2.0
            elif epoch < 30:
                return 1.0
            elif epoch < 40:
                return 0.5
            else:
                return 0.1

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        train_metrics = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            start_time = time()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Learning Rate: {current_lr:.6f}")

            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc="Training"):
                images = batch["x"].to(device)
                labels = batch["y"].to(device)
                labels = labels.unsqueeze(1).float()

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                correct += (predicted == labels.int()).sum().item()
                total += labels.size(0)

            avg_train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            self.eval()
            valid_running_loss = 0.0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validation"):
                    images = batch["x"].to(device)
                    labels = batch["y"].to(device)
                    labels = labels.unsqueeze(1).float()

                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_running_loss += loss.item()

                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    valid_correct += (predicted == labels.int()).sum().item()
                    valid_total += labels.size(0)

            avg_valid_loss = valid_running_loss / len(valid_loader)
            valid_acc = 100 * valid_correct / valid_total


            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                best_model_state = self.state_dict()  

            train_metrics['epochs'].append(epoch + 1)
            train_metrics['train_loss'].append(avg_train_loss)
            train_metrics['train_acc'].append(train_acc)
            train_metrics['val_loss'].append(avg_valid_loss)
            train_metrics['val_acc'].append(valid_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"| Train Acc: {train_acc:.2f}% "
                f"| Valid Acc: {valid_acc:.2f}% "
                f"| Valid Loss: {avg_valid_loss:.4f} "
                f"| Time: {time() - start_time:.2f}s")

            scheduler.step()

        self.load_state_dict(best_model_state)
        
        return train_metrics


    def test_legib(self, test_loader=None, device='cpu', average='macro'):
        self.to(device)
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["x"].to(device)
                labels = batch["y"].to(device)

                outputs = self(images)
                preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, average=average, zero_division=0)
        prec = precision_score(all_labels, all_preds, average=average, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Accuracy:  {acc:.4f}\n")
        print(f"Precision: {prec:.4f}\n")
        print(f"Recall:    {rec:.4f}\n")
        print(f"F1-score:  {f1:.4f}\n")

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        plt.show()
        plt.close(fig)

        return {"acc": acc,"precision": prec,"recall": rec,"f1": f1} , fig
    

def grid_filtre(model, test_loader, device="cuda", num_images=100, filtre="all"):
    model.to(device)
    model.eval()

    images_to_show = []
    preds_to_show = []
    labels_to_show = []
    probs_to_show = []  

    with torch.no_grad():
        for batch in test_loader:
            images = batch["x"].to(device)
            labels = batch["y"].to(device)
            output = model(images)

            probs = torch.sigmoid(output).squeeze(1)  
            preds = (probs > 0.5).int()

            for i in range(images.size(0)):
                prob = probs[i].item()
                pred = preds[i].item()
                label = labels[i].item()

                if filtre == "correct" and pred != label:
                    continue
                elif filtre == "wrong" and pred == label:
                    continue

                images_to_show.append(images[i].cpu())
                preds_to_show.append(pred)
                labels_to_show.append(label)
                probs_to_show.append(prob)  

                if len(images_to_show) >= num_images:
                    break
            if len(images_to_show) >= num_images:
                break

    # --- Affichage en grille ---
    cols = 10
    rows = (num_images + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))
    num_display = min(num_images, len(images_to_show))

    for i in range(num_display):
        img = images_to_show[i]
        pred = preds_to_show[i]
        label = labels_to_show[i]
        prob = probs_to_show[i]

        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        color = "green" if pred == label else "red"
        title = f"True: {label}\nPred: {pred} ({prob:.2f})" 

        ax.text(
            5, 20,
            title,
            fontsize=10,
            color=color,
            bbox=dict(facecolor="black", alpha=0.6, pad=3)
        )

    plt.tight_layout()
    return fig
    