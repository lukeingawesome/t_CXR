# %%
import wandb
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from health_multimodal.image.model.modules import MultiTaskModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from torch.optim.lr_scheduler import LambdaLR
import argparse
import pandas as pd
from health_multimodal.Mdataset import MimicDataset
# %%
parser = argparse.ArgumentParser(description="Example script with command-line options")
parser.add_argument("--freeze", "-f", action="store_true", help="Enable verbose mode")
parser.add_argument("--cuda", "-cuda", type=int, default=0, help="cuda number be specified")
parser.add_argument("--findings", "-findings", type=str, default="Pneumonia", help="findings be specified")

args = parser.parse_args()
# %%

if args.freeze:
    project_name = f"annealing500-kl-new-six-freeze-{args.findings}"
else:
    project_name = f"annealing500-kl-new-six-{args.findings}"

wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "BIOVIL-T",
    "dataset": "",
    "epochs": 30,
    }
)

# %%
model=get_biovil_t_image_encoder()

# %%
# model.feature_size:512
# map = {"absent": 0, "new": 1, "worsened": 2,"stable": 3,"improved": 4, "resolved": 5}
model.classifier= MultiTaskModel(input_dim = model.feature_size*2 ,classifier_hidden_dim=None, num_classes=6, num_tasks=1)

# %%
criterion = nn.CrossEntropyLoss()
kl_criterion = nn.KLDivLoss(reduction="batchmean")
base_lr = 2e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

# %%
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
train_dataset= MimicDataset(csv_path ="/opt/project/t_CXR/src/mimic_interval_org_prep.csv", findings=f"{args.findings}", is_test= False)
test_dataset = MimicDataset(csv_path ="/opt/project/t_CXR/src/mimic_interval_org_prep.csv", findings=f"{args.findings}", is_test= True)

# 10% test
train_size = int(0.9 * len(train_dataset))  # 90 * 80% 훈련
val_size = len(train_dataset) - train_size  # 10% 검증

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=30, shuffle=True)
# %%
num_epochs = 30

num_training_steps = len(train_loader) * num_epochs  # 총 학습 스텝 (50 epochs)
warmup_steps = int(num_training_steps * 0.03)  # 3% Warmup
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)  # Warmup 단계
    return max(0.0, float(num_training_steps - current_step) / float(num_training_steps - warmup_steps))  # Linear Decay

scheduler = LambdaLR(optimizer, lr_lambda)

# %%
model.to(device)
model.freeze_encoder = True
# %%
best_val_loss = float("inf")
# %%
# map = {"absent": 0, "new": 1, "worsened": 2,"stable": 3,"improved": 4, "resolved": 5}
reverse_label_mapping = torch.tensor([0, 5, 4, 3, 2, 1]) # index와 value를 이용해서 O(1) 연산
reverse_label_mapping= reverse_label_mapping.to(device)

avg_train_loss = float("1")
avg_train_entropy_loss = float("1")
avg_train_kl_loss = float("1")

def sigmoid_annealing(epoch, midpoint, steepness=5):
    return 1 / (1 + np.exp(-steepness * (epoch - midpoint)))

kl_weight = 500.0

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    lambda_kl = sigmoid_annealing(epoch, 10, steepness=7) # 30 epoch 중 10 epoch부터 시작하도록

    total_train_loss = 0.0
    train_kl_loss = 0.0
    train_entropy_loss = 0.0

    for curr_data, prev_data, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False, mininterval=10):
        # Forward
        curr_data, prev_data, labels = curr_data.to(device), prev_data.to(device), labels.to(device)
        labels= labels.long()
        optimizer.zero_grad()

        # cross entropy (curr, prev)
        outputs = model(curr_data, prev_data).class_logits.squeeze(-1)
        cross_entropy_loss = criterion(outputs, labels)
        # cross entropy (prev,curr)
        reveresed_outputs = model(prev_data, curr_data).class_logits.squeeze(-1)
        reversed_label = reverse_label_mapping[labels]
        reversed_cross_entropy_loss = criterion(reveresed_outputs,reversed_label)
        # kl loss
        prob = torch.log_softmax(outputs , dim=-1)
        reversed_prob = torch.softmax(reveresed_outputs, dim=-1)
        swapped_reversed_prob = reversed_prob.clone().detach()
        swapped_reversed_prob[:, [1, 5]] = reversed_prob[:, [5, 1]]  # ✅ 안전한 out-of-place 연산
        swapped_reversed_prob[:, [2, 4]] = reversed_prob[:, [4, 2]]
        kl_loss = kl_criterion(prob, swapped_reversed_prob)

        # total loss = cross+cross+kl
        # loss = cross_entropy_loss+ reversed_cross_entropy_loss +  kl_loss * kl_weight 
        loss = cross_entropy_loss+ reversed_cross_entropy_loss +  kl_loss * lambda_kl * kl_weight 

        #
        train_entropy_loss += cross_entropy_loss.item() + reversed_cross_entropy_loss.item()
        train_kl_loss += kl_loss.item()
        total_train_loss += loss.item()

        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()  # 평가 모드 설정
    total_val_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():  # ✅ [5] 검증 과정에서는 그래디언트 업데이트 X
        for curr_data, prev_data, labels in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}", leave=False, mininterval=10):  # ✅ [6] Validation Loop 시작
            curr_data, prev_data, labels = curr_data.to(device), prev_data.to(device), labels.to(device)

            # cross entropy (curr, prev)
            outputs = model(curr_data, prev_data).class_logits.squeeze(-1)
            cross_entropy_loss = criterion(outputs, labels)
            # cross entropy (prev,curr)
            reveresed_outputs = model(prev_data, curr_data).class_logits.squeeze(-1)
            reversed_label = reverse_label_mapping[labels].to(device)
            reversed_cross_entropy_loss = criterion(reveresed_outputs,reversed_label)
            # kl loss
            prob = torch.log_softmax(outputs , dim=-1)
            reversed_prob = torch.softmax(reveresed_outputs, dim=-1)
            swapped_reversed_prob = reversed_prob.clone()
            swapped_reversed_prob[:, [1, 5]] = reversed_prob[:, [5, 1]]  # ✅ 안전한 out-of-place 연산
            swapped_reversed_prob[:, [2, 4]] = reversed_prob[:, [4, 2]]

            kl_loss = kl_criterion(prob, swapped_reversed_prob)
            # total loss = cross+cross+kl
            loss = cross_entropy_loss+ reversed_cross_entropy_loss + lambda_kl * kl_weight * kl_loss

            total_val_loss += loss.item()

            prob = torch.softmax(outputs , dim=-1)
            all_labels.append(labels)
            all_probs.append(prob)

    epoch_time = time.time() - start_time

    all_labels = torch.cat(all_labels, dim =0).cpu()
    all_probs = torch.cat(all_probs, dim=0).cpu()
    all_preds = np.argmax(all_probs, axis=1)

    mic_auroc = roc_auc_score(all_labels, all_probs, average='micro', multi_class="ovr")
    mac_auroc = roc_auc_score(all_labels, all_probs, average='macro', multi_class="ovr")
    overall_accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_entropy_loss = train_entropy_loss/len(train_loader)
    avg_train_kl_loss = lambda_kl * kl_weight * train_kl_loss/len(train_loader)

    avg_val_loss = total_val_loss / len(valid_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MIC_AUROC: {mic_auroc:.4f} | MAC_AUROC: {mac_auroc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Micro: {f1_micro:.4f} | precision_macro: {precision_macro} | recall_macro: {recall_macro} | precision_micro: f{precision_micro} | recall_micro: {recall_micro} | Time: {epoch_time:.2f}s")
    value_counts = torch.bincount(all_preds)
    df = pd.DataFrame({
        "Value": torch.arange(value_counts.size(0)),
        "Frequency": value_counts.numpy()
    })

    wandb.log({
            "Train Loss": avg_train_loss,
            "Train Entropy Loss ":avg_train_entropy_loss,
            "Train KL Loss ":avg_train_kl_loss,
            "Val Loss": avg_val_loss,
            "overall_accuracy": overall_accuracy,
            "MIC_AUROC": mic_auroc ,
            "MAC_AUROC": mac_auroc,
            "F1-Macro": f1_macro,
            "F1-Micro": f1_micro ,
            "Time": epoch_time,
           "precision_macro": precision_macro,
           "recall_macro": recall_macro,
           "precision_micro": precision_micro,
           "recall_micro": recall_micro,
           "Frequency": wandb.Table(dataframe=df)
        })

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # save_checkpoint(epoch, model, optimizer, avg_val_loss, filename=f"annealing500-kl-new-six_{args.findings}_best_checkpoint.pth", rank=0)

# %%
wandb.finish()
