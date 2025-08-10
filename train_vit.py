import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ================================
# データセット
# ================================
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.encoder = LabelEncoder()
        self.data['label'] = self.encoder.fit_transform(self.data['label'])
        self.classes = self.encoder.classes_
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ================================
# 前処理
# ================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

transform_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

train_dataset = ImageDataset("train.csv", transform=transform_train)
valid_dataset = ImageDataset("valid.csv", transform=transform_valid)

# batchサイズは軽めにしてメモリ不可を抑える（batch_size=8～16）
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# ================================
# モデル
# ================================
device = torch.device("cpu")
# vit_base_patch16_224 ではなく vit_small_patch16_224の方が学習が早くなる
model = timm.create_model("vit_small_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# ================================
# 学習ループ
# ================================
EPOCHS = 5
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
    
    # ベストモデル保存
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': train_dataset.classes
        }, "vit_model.pth")
        print(f"✅ Best model saved (Acc: {best_acc:.4f})")

print("🎉 Training completed! vit_model.pth saved.")
