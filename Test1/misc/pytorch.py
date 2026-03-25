# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models
# from sklearn.metrics import roc_auc_score
# from torch.nn.functional import softmax

# # ─── Device Setup ────────────────────────────────────────────────────────────
# try:
#     import torch_directml
#     device = torch_directml.device()
#     print(f"Using device: {device}")
# except Exception:
#     device = torch.device('cpu')
#     print("Using device: CPU")

# # ─── Config ──────────────────────────────────────────────────────────────────
# folder_path = r"../dataset"
# types       = ['no', 'sphere', 'vort']
# label_map   = {'no': 0, 'sphere': 1, 'vort': 2}

# width       = 128
# num_epochs  = 30
# batch_size  = 128
# lr          = 1e-4
# num_classes = 3

# # ─── Data Loading ─────────────────────────────────────────────────────────────
# def load_data(split):
#     data, labels = [], []
#     for t in types:
#         path = os.path.join(folder_path, split, t)
#         for npy_name in os.listdir(path):
#             arr = np.load(os.path.join(path, npy_name))   # shape: (C, H, W) or (H, W)
#             data.append(arr)
#             labels.append(label_map[t])
#     return np.array(data), np.array(labels)

# X_train, Y_train = load_data('train')
# X_val,   Y_val   = load_data('val')

# # Original code did (0,2,3,1) → NCHW→NHWC for Keras.
# # PyTorch expects NCHW, so we keep the original axis order.
# # If your .npy files are already (H, W) 2-D arrays, we add a channel dim.
# if X_train.ndim == 3:          # (N, H, W) → (N, 1, H, W)
#     X_train = X_train[:, None, :, :]
#     X_val   = X_val[:,   None, :, :]
# elif X_train.ndim == 4:
#     # Files were stored as (N, C, H, W) – keep as-is
#     # If they were (N, H, W, C) after the Keras transpose, undo it:
#     if X_train.shape[-1] in (1, 3):   # last dim looks like channels
#         X_train = X_train.transpose(0, 3, 1, 2)
#         X_val   = X_val.transpose(0, 3, 1, 2)

# X_train = X_train.astype(np.float32)
# X_val   = X_val.astype(np.float32)

# print(f"Train shape : {X_train.shape},  Labels: {Y_train.shape}")
# print(f"Val   shape : {X_val.shape},    Labels: {Y_val.shape}")

# # ─── Dataset & DataLoader ─────────────────────────────────────────────────────
# class NpyDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = torch.from_numpy(X)
#         self.Y = torch.from_numpy(Y).long()

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]

# train_loader = DataLoader(NpyDataset(X_train, Y_train),
#                           batch_size=batch_size, shuffle=True,  num_workers=0)
# val_loader   = DataLoader(NpyDataset(X_val,   Y_val),
#                           batch_size=batch_size, shuffle=False, num_workers=0)

# # ─── DirectML-safe norm helpers ───────────────────────────────────────────────
# # BatchNorm (both 1-D and 2-D) is unsupported on DirectML.
# # GroupNorm is a drop-in replacement that works on all backends.

# def gn(num_channels, num_groups=32):
#     """GroupNorm that gracefully falls back to fewer groups if needed."""
#     # num_groups must divide num_channels
#     while num_groups > 1 and num_channels % num_groups != 0:
#         num_groups //= 2
#     return nn.GroupNorm(num_groups, num_channels)

# # GroupNorm factory that ResNet's norm_layer kwarg expects:
# # it receives only (num_channels,) so we wrap our gn() helper.
# def gn_layer(num_channels):
#     num_groups = 32
#     while num_groups > 1 and num_channels % num_groups != 0:
#         num_groups //= 2
#     return nn.GroupNorm(num_groups, num_channels)

# def verify_no_bn(module):
#     """Scan the full module tree — raises if any BatchNorm layer remains."""
#     found = [name for name, m in module.named_modules()
#              if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm))]
#     if found:
#         raise RuntimeError(f"BatchNorm still present at: {found}")
#     print("✓ No BatchNorm layers — safe for DirectML.")

# # ─── Model ────────────────────────────────────────────────────────────────────
# class SupervisedModel(nn.Module):
#     def __init__(self, width=128, num_classes=3):
#         super().__init__()

#         # Build ResNet50 with GroupNorm from the start via norm_layer kwarg.
#         # This means BatchNorm is NEVER instantiated anywhere in the backbone.
#         backbone = models.resnet50(weights=None, norm_layer=gn_layer)
#         # 1-channel input (grayscale)
#         backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         in_features = backbone.fc.in_features   # 2048
#         backbone.fc  = nn.Identity()
#         self.backbone = backbone

#         # Classifier head — NO norm layers at all (backbone GN is sufficient)
#         self.head = nn.Sequential(
#             # first block
#             nn.Linear(in_features, width * 8),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             # second block
#             nn.Linear(width * 8, width * 4),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             # third block
#             nn.Linear(width * 4, width),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             # output
#             nn.Linear(width, num_classes),
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         return self.head(x)           # raw logits

# model = SupervisedModel(width=width, num_classes=num_classes)
# verify_no_bn(model)       # prints ✓ or raises clearly before touching the GPU
# model = model.to(device)
# print(model)

# # ─── Loss, Optimiser ──────────────────────────────────────────────────────────
# criterion = nn.CrossEntropyLoss()          # expects raw logits + integer labels
# optimizer = optim.Adam(model.parameters(), lr=lr)

# # ─── Training Loop ────────────────────────────────────────────────────────────
# history = {'loss': [], 'acc': [], 'auc': [],
#            'val_loss': [], 'val_acc': [], 'val_auc': []}

# def run_epoch(loader, training=True):
#     if training:
#         model.train()
#     else:
#         model.eval()

#     total_loss, correct, total = 0.0, 0, 0
#     all_probs, all_labels = [], []

#     ctx = torch.enable_grad() if training else torch.no_grad()
#     with ctx:
#         for X_batch, Y_batch in loader:
#             X_batch = X_batch.to(device)
#             Y_batch = Y_batch.to(device)

#             logits = model(X_batch)
#             loss   = criterion(logits, Y_batch)

#             if training:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             total_loss += loss.item() * len(Y_batch)
#             preds       = logits.argmax(dim=1)
#             correct    += (preds == Y_batch).sum().item()
#             total      += len(Y_batch)

#             probs = softmax(logits, dim=1).cpu().numpy()
#             all_probs.append(probs)
#             all_labels.append(Y_batch.cpu().numpy())

#     avg_loss = total_loss / total
#     acc      = correct / total
#     all_probs  = np.concatenate(all_probs,  axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)

#     # One-vs-rest macro AUC (matches Keras AUC behaviour)
#     try:
#         from sklearn.preprocessing import label_binarize
#         Y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
#         auc   = roc_auc_score(Y_bin, all_probs, multi_class='ovr', average='macro')
#     except Exception:
#         auc = float('nan')

#     return avg_loss, acc, auc


# for epoch in range(1, num_epochs + 1):
#     tr_loss, tr_acc, tr_auc = run_epoch(train_loader, training=True)
#     vl_loss, vl_acc, vl_auc = run_epoch(val_loader,   training=False)

#     history['loss'].append(tr_loss);     history['val_loss'].append(vl_loss)
#     history['acc'].append(tr_acc);       history['val_acc'].append(vl_acc)
#     history['auc'].append(tr_auc);       history['val_auc'].append(vl_auc)

#     print(f"Epoch {epoch:3d}/{num_epochs} | "
#           f"loss {tr_loss:.4f}  acc {tr_acc:.4f}  auc {tr_auc:.4f} | "
#           f"val_loss {vl_loss:.4f}  val_acc {vl_acc:.4f}  val_auc {vl_auc:.4f}")

# print("Training complete.")

# # ─── Save Model ───────────────────────────────────────────────────────────────
# torch.save(model.state_dict(), "model_supervised.pth")
# print("Model saved to model_supervised.pth")



import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

# ─── Device Setup ─────────────────────────────────────────────────────────────
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using device: {device}")
except Exception:
    device = torch.device('cpu')
    print("Using device: CPU")

# ─── Config ───────────────────────────────────────────────────────────────────
folder_path = r"../dataset"
types       = ['no', 'sphere', 'vort']
label_map   = {'no': 0, 'sphere': 1, 'vort': 2}

width       = 128
num_epochs  = 30
batch_size  = 128
lr          = 1e-4
num_classes = 3

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(split):
    data, labels = [], []
    for t in types:
        path = os.path.join(folder_path, split, t)
        for npy_name in os.listdir(path):
            arr = np.load(os.path.join(path, npy_name))
            data.append(arr)
            labels.append(label_map[t])
    return np.array(data), np.array(labels)

X_train, Y_train = load_data('train')
X_val,   Y_val   = load_data('val')

if X_train.ndim == 3:
    X_train = X_train[:, None, :, :]
    X_val   = X_val[:,   None, :, :]
elif X_train.ndim == 4 and X_train.shape[-1] in (1, 3):
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val   = X_val.transpose(0, 3, 1, 2)

X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)

print(f"Train : {X_train.shape}  |  Val : {X_val.shape}")

# ─── Dataset & DataLoader ─────────────────────────────────────────────────────
class NpyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y).long()
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

train_loader = DataLoader(NpyDataset(X_train, Y_train),
                          batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(NpyDataset(X_val,   Y_val),
                          batch_size=batch_size, shuffle=False, num_workers=0)

# ─── Remove ALL norm layers from a module (DirectML has no norm kernels) ──────
def remove_all_norms(module: nn.Module) -> nn.Module:
    """
    Walk the entire module tree and replace every norm layer with nn.Identity().
    Works on the live object in-place — no re-instantiation needed.
    Covers: BatchNorm1d/2d/3d, SyncBatchNorm, GroupNorm, LayerNorm,
            InstanceNorm1d/2d/3d.
    """
    NORM_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, NORM_TYPES):
                setattr(parent, name, nn.Identity())
    return module

# ─── Model ────────────────────────────────────────────────────────────────────
class SupervisedModel(nn.Module):
    def __init__(self, width: int = 128, num_classes: int = 3):
        super().__init__()

        # Standard ResNet50, no norm_layer override needed
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Classifier head — no norm layers
        self.head = nn.Sequential(
            nn.Linear(in_features, width * 8),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(width * 8, width * 4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(width * 4, width),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(width, num_classes),
        )

        # Strip every norm layer from the whole model after construction
        remove_all_norms(self)

    def forward(self, x):
        return self.head(self.backbone(x))

model = SupervisedModel(width=width, num_classes=num_classes).to(device)

# Verify zero norm layers remain
NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
              nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
              nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
remaining = [n for n, m in model.named_modules() if isinstance(m, NORM_TYPES)]
if remaining:
    print(f"WARNING — norm layers still present: {remaining}")
else:
    print("✓ Zero norm layers — fully safe for DirectML.")

# ─── Loss & Optimiser ─────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ─── Training Loop ────────────────────────────────────────────────────────────
history = {'loss': [], 'acc': [], 'auc': [],
           'val_loss': [], 'val_acc': [], 'val_auc': []}

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, Y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(Y_batch)
            correct    += (logits.argmax(1) == Y_batch).sum().item()
            total      += len(Y_batch)

            all_probs.append(softmax(logits, dim=1).cpu().numpy())
            all_labels.append(Y_batch.cpu().numpy())

    avg_loss   = total_loss / total
    acc        = correct / total
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    try:
        from sklearn.preprocessing import label_binarize
        Y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        auc   = roc_auc_score(Y_bin, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    return avg_loss, acc, auc


for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_auc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc, vl_auc = run_epoch(val_loader,   training=False)

    history['loss'].append(tr_loss);     history['val_loss'].append(vl_loss)
    history['acc'].append(tr_acc);       history['val_acc'].append(vl_acc)
    history['auc'].append(tr_auc);       history['val_auc'].append(vl_auc)

    print(f"Epoch {epoch:3d}/{num_epochs} | "
          f"loss {tr_loss:.4f}  acc {tr_acc:.4f}  auc {tr_auc:.4f} | "
          f"val_loss {vl_loss:.4f}  val_acc {vl_acc:.4f}  val_auc {vl_auc:.4f}")

print("Training complete.")
torch.save(model.state_dict(), "model_supervised.pth")
print("Model saved → model_supervised.pth")