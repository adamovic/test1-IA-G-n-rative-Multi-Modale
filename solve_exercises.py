import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("=== EXERCICE 1 : REGRESSION ===")
torch.manual_seed(42)
n = 300
anciennete = torch.randint(0, 21, (n, 1)).float()
expertise = torch.randint(1, 6, (n, 1)).float()
bruit = torch.randn(n, 1) * 500
salaires = 4000 + 600 * anciennete + 1500 * expertise + bruit
X = torch.cat([anciennete, expertise], dim=1)
y = salaires

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print(f"Salaire moyen : {y.mean().item():.2f} DH")

# Train / Test split
indices = torch.randperm(n)
limite = int(0.8 * n)
train_indices = indices[:limite]
test_indices = indices[limite:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices] / 1000.0, y[test_indices] / 1000.0

moyenne = X_train.mean(dim=0, keepdim=True)
ecart_type = X_train.std(dim=0, keepdim=True)

X_train_norm = (X_train - moyenne) / ecart_type
X_test_norm = (X_test - moyenne) / ecart_type

# Modèle régression
model_reg = nn.Linear(2, 1)
criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.SGD(model_reg.parameters(), lr=0.05)

historique_reg = []
model_reg.train()
for epoch in range(500):
    optimizer_reg.zero_grad()
    predictions = model_reg(X_train_norm)
    loss = criterion_reg(predictions, y_train)
    loss.backward()
    optimizer_reg.step()
    historique_reg.append(loss.item())

print(f"Perte finale entraînement (kDH^2) : {historique_reg[-1]:.4f}")

# Evaluation MAE
model_reg.eval()
with torch.no_grad():
    predictions_test = model_reg(X_test_norm)
    mae = torch.mean(torch.abs(predictions_test * 1000.0 - y_test * 1000.0))
print(f"MAE test : {mae.item():.2f} DH")

# Nouveau salarié : 7 ans, expertise 4
nouveau_salarie = torch.tensor([[7.0, 4.0]])
nouveau_salarie_norm = (nouveau_salarie - moyenne) / ecart_type
with torch.no_grad():
    salaire_predit = model_reg(nouveau_salarie_norm) * 1000.0
print(f"Salaire estimé (7 ans, exp 4) : {salaire_predit.item():.2f} DH (Théorique sans bruit : 14200.00 DH)")

# Coefficients dé-normalisés
w = model_reg.weight.data.squeeze() * 1000.0
b = model_reg.bias.data.item() * 1000.0
mu = moyenne.squeeze()
sigma = ecart_type.squeeze()
w1_retrouve = (w[0] / sigma[0]).item()
w2_retrouve = (w[1] / sigma[1]).item()
b_retrouve = (b - (w[0] * mu[0] / sigma[0]) - (w[1] * mu[1] / sigma[1])).item()
print(f"Coefficients retrouvés : w1={w1_retrouve:.2f} (cible 600), w2={w2_retrouve:.2f} (cible 1500), b={b_retrouve:.2f} (cible 4000)")


print("\n=== EXERCICE 2 : CLASSIFICATION IRIS ===")
torch.manual_seed(42)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)
scaler = StandardScaler()
X_train_i = scaler.fit_transform(X_train_i)
X_test_i = scaler.transform(X_test_i)

X_train_i = torch.tensor(X_train_i, dtype=torch.float32)
X_test_i = torch.tensor(X_test_i, dtype=torch.float32)
y_train_i = torch.tensor(y_train_i, dtype=torch.long)
y_test_i = torch.tensor(y_test_i, dtype=torch.long)

model_iris = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
criterion_iris = nn.CrossEntropyLoss()
optimizer_iris = torch.optim.Adam(model_iris.parameters(), lr=0.01)

losses_iris = []
model_iris.train()
for epoch in range(200):
    optimizer_iris.zero_grad()
    logits = model_iris(X_train_i)
    loss = criterion_iris(logits, y_train_i)
    loss.backward()
    optimizer_iris.step()
    losses_iris.append(loss.item())

model_iris.eval()
with torch.no_grad():
    logits_test = model_iris(X_test_i)
    preds_iris = logits_test.argmax(dim=1)
    acc_iris = (preds_iris == y_test_i).float().mean()
print(f"Accuracy Iris Test : {acc_iris.item():.1%}")

# Matrice de confusion
cm_iris = confusion_matrix(y_test_i.numpy(), preds_iris.numpy())
print("Matrice de confusion Iris :\n", cm_iris)

# Nouvelle fleur
nouvelle_fleur = np.array([[5.1, 3.5, 1.4, 0.2]])
nouvelle_fleur_norm = torch.tensor(scaler.transform(nouvelle_fleur), dtype=torch.float32)
with torch.no_grad():
    pred_espece = model_iris(nouvelle_fleur_norm).argmax(dim=1).item()
print(f"Prédiction nouvelle fleur [5.1, 3.5, 1.4, 0.2] : classe {pred_espece} ({iris.target_names[pred_espece]})")


print("\n=== EXERCICE 3 : MLP MNIST ===")
torch.manual_seed(42)
transform_mnist = transforms.ToTensor()
train_dataset_mlp = datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
test_dataset_mlp = datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)

train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=64, shuffle=True)
test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=256, shuffle=False)

model_mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.001)

# Entraînement 5 époques
epochs_mlp = 5
for epoch in range(epochs_mlp):
    model_mlp.train()
    total_loss = 0.0
    for imgs, lbls in train_loader_mlp:
        optimizer_mlp.zero_grad()
        outputs = model_mlp(imgs)
        loss = criterion_mlp(outputs, lbls)
        loss.backward()
        optimizer_mlp.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"MLP Epoch {epoch+1}/{epochs_mlp} | Loss: {total_loss / len(train_dataset_mlp):.4f}")

# Evaluation
model_mlp.eval()
all_preds_mlp = []
all_labels_mlp = []
with torch.no_grad():
    for imgs, lbls in test_loader_mlp:
        outputs = model_mlp(imgs)
        preds = outputs.argmax(dim=1)
        all_preds_mlp.extend(preds.numpy())
        all_labels_mlp.extend(lbls.numpy())

all_preds_mlp = np.array(all_preds_mlp)
all_labels_mlp = np.array(all_labels_mlp)

acc_mlp = accuracy_score(all_labels_mlp, all_preds_mlp)
prec_mlp = precision_score(all_labels_mlp, all_preds_mlp, average='weighted')
rec_mlp = recall_score(all_labels_mlp, all_preds_mlp, average='weighted')
f1_mlp = f1_score(all_labels_mlp, all_preds_mlp, average='weighted')

print(f"MLP Test Metrics -> Accuracy: {acc_mlp:.4f}, Precision: {prec_mlp:.4f}, Recall: {rec_mlp:.4f}, F1-score: {f1_mlp:.4f}")

# Sauvegarde et rechargement
torch.save(model_mlp.state_dict(), "mnist_mlp.pth")
model_reloaded = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model_reloaded.load_state_dict(torch.load("mnist_mlp.pth", weights_only=True))
model_reloaded.eval()
print("Modèle MLP sauvegardé et rechargé avec succès.")


print("\n=== EXERCICE 4 : CNN MNIST ===")
torch.manual_seed(42)
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset_cnn = datasets.MNIST(root="./data", train=True, download=True, transform=transform_cnn)
test_dataset_cnn = datasets.MNIST(root="./data", train=False, download=True, transform=transform_cnn)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=256, shuffle=False)

class MNISTClassifier(nn.Module):
    def __init__(self, with_dropout=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU()
        ]
        if with_dropout:
            classifier_layers.append(nn.Dropout(p=0.5))
        classifier_layers.append(nn.Linear(128, 10))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model_cnn = MNISTClassifier(with_dropout=False)
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=0.001)

epochs_cnn = 3
for epoch in range(epochs_cnn):
    model_cnn.train()
    total_loss = 0.0
    for imgs, lbls in train_loader_cnn:
        optimizer_cnn.zero_grad()
        logits = model_cnn(imgs)
        loss = criterion_cnn(logits, lbls)
        loss.backward()
        optimizer_cnn.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"CNN Epoch {epoch+1}/{epochs_cnn} | Loss: {total_loss / len(train_dataset_cnn):.4f}")

model_cnn.eval()
correct = 0
total = 0
misclassified_imgs = []
misclassified_true = []
misclassified_pred = []

with torch.no_grad():
    for imgs, lbls in test_loader_cnn:
        logits = model_cnn(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        
        # Collecter les erreurs
        mask = (preds != lbls)
        if mask.any() and len(misclassified_imgs) < 10:
            for img, t, p in zip(imgs[mask], lbls[mask], preds[mask]):
                misclassified_imgs.append(img)
                misclassified_true.append(t.item())
                misclassified_pred.append(p.item())
                if len(misclassified_imgs) >= 10:
                    break

acc_cnn = correct / total
print(f"CNN Test Accuracy : {acc_cnn:.2%}")
print(f"Nombre d'images mal classées trouvées : {len(misclassified_imgs)}")
for idx in range(min(5, len(misclassified_imgs))):
    print(f"  Erreur {idx+1}: Vrai={misclassified_true[idx]}, Prédit={misclassified_pred[idx]}")

torch.save(model_cnn.state_dict(), "mnist_cnn.pth")
print("Modèle CNN sauvegardé dans mnist_cnn.pth.")
print("\n=== TOUT S'EST DEROULE AVEC SUCCES ===")
