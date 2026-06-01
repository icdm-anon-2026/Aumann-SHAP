import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# 1) Load MNIST
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

print("Full MNIST sizes:", len(train), len(test))

# -----------------------------
# 2) Filter to only digits 1 and 7
# -----------------------------
idx_train = [i for i in range(len(train)) if train[i][1] in (1, 7)]
idx_test  = [i for i in range(len(test))  if test[i][1] in (1, 7)]

train_17 = Subset(train, idx_train)
test_17  = Subset(test, idx_test)

print("Filtered (1 & 7) sizes:", len(train_17), len(test_17))

# -----------------------------
# 3) DataLoaders
# -----------------------------
train_loader = DataLoader(train_17, batch_size=128, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_17, batch_size=256, shuffle=False, num_workers=0)

# -----------------------------
# 4) Build ResNet-18 for MNIST (1-channel, 28x28, 2 classes)
# -----------------------------
model = resnet18(weights=None)

# change first conv to accept 1 channel and not downsample too early
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # remove maxpool for small images

# change classifier head to 2 classes (1 vs 7)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)
print(model)

# -----------------------------
# 5) Train + Eval helpers
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = (y == 7).long().to(device)  # 1->0, 7->1

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total

# -----------------------------
# 6) Train loop + save best checkpoint
# -----------------------------
epochs = 5
lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
}

best_test_acc = -1.0
save_path = "resnet18_mnist_1vs7.pt"

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in train_loader:
        x = x.to(device)
        y = (y == 7).long().to(device)  # 1->0, 7->1

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    print(f"Epoch {epoch}/{epochs} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    # save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_test_acc": best_test_acc,
            "history": history,
        }, save_path)
        print(f"Saved best checkpoint -> {save_path} (acc={best_test_acc:.4f})")

print("Done. Best test acc:", best_test_acc)

# -----------------------------
# 7) quick plot of training curves
# -----------------------------
plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["test_loss"], label="test_loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["test_acc"], label="test_acc")
plt.legend()
plt.title("Accuracy")
plt.show()

