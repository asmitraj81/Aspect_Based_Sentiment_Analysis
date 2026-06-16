import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

pip install transformers accelerate
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os

# ===============================
# Parse MAMS XML
# ===============================
import os

print("Train exists:", os.path.exists("train.xml"))
print("Test exists:", os.path.exists("test.xml"))
print("Train size:", os.path.getsize("train.xml") if os.path.exists("train.xml") else "No file")
def parse_mams_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for sentence in root.findall("sentence"):
        text = sentence.find("text").text.strip()
        aspects = sentence.find("aspectCategories")

        if aspects is not None:
            for aspect in aspects.findall("aspectCategory"):
                data.append({
                    "text": text,
                    "aspect": aspect.get("category"),
                    "sentiment": aspect.get("polarity")
                })

    return pd.DataFrame(data)

train_df = parse_mams_xml("train.xml")
test_df = parse_mams_xml("test.xml")

# ===============================
# Encode Labels
# ===============================

label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["sentiment"])
test_df["label"] = label_encoder.transform(test_df["sentiment"])

num_labels = len(label_encoder.classes_)

# ===============================
# Dataset Class
# ===============================

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["aspect"],
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }

# ===============================
# Setup
# ===============================

device = torch.device("cuda")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
).to(device)

train_dataset = ABSADataset(train_df, tokenizer)
test_dataset = ABSADataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ===============================
# Class Weights
# ===============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * 5

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler()

# ===============================
# Training Loop
# ===============================

best_f1 = 0
patience = 2
trigger_times = 0

epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", total_loss / len(train_loader))

    # ===============================
    # Evaluation
    # ===============================

    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(true_labels, preds, average="macro")

    print("Macro F1:", macro_f1)
    print(classification_report(true_labels, preds))

    # ===============================
    # Early Stopping
    # ===============================

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
        print("Best model saved.")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

print("Best Macro F1 Achieved:", best_f1)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
!pip install transformers
# -------------------------------
# Generate Predictions
# -------------------------------

model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(true_labels, preds)
accuracy = accuracy_score(true_labels, preds)

print("\n========== FINAL MODEL PERFORMANCE ==========")
print(f"Overall Accuracy: {accuracy:.4f}\n")
print(classification_report(true_labels, preds))

epochs = [1,2,3,4,5]
train_losses = [1.0517, 0.7658, 0.6074, 0.5022, 0.4293]
macro_f1_scores = [0.6235, 0.7497, 0.7627, 0.7824, 0.7924]

# ==========================================================
# 1️⃣ Line Plot – Training Loss
# ==========================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_losses,
    marker='o',
    linestyle='-',
    linewidth=2.5,
    markersize=7,
    color='royalblue',
    label="Training Loss"
)

plt.title("Training Loss Trend", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()

# ==========================================================
# 2️⃣ Line Plot – Macro F1 Score
# ==========================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    macro_f1_scores,
    marker='o',
    markersize=8,
    linewidth=3,
    color='#2E86C1',
    label="Macro F1 Score"
)

# Highlight improvement area
plt.fill_between(
    epochs,
    macro_f1_scores,
    color='#85C1E9',
    alpha=0.3
)

plt.title("Macro F1 Score Improvement Across Epochs",
          fontsize=16,
          fontweight='bold')

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Macro F1 Score", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()

# ==========================================================
# 3️⃣ Heatmap – Confusion Matrix
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

# Example confusion matrix (replace with your cm)
cm = np.array([[50, 2, 1],
               [4, 45, 3],
               [2, 1, 48]])

plt.figure(figsize=(6,5))

# Use distinct colors
cmap = plt.cm.tab20

# Unique color index for each cell
color_index = np.arange(cm.size).reshape(cm.shape)

img = plt.imshow(color_index, cmap=cmap)

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Show actual confusion matrix values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha='center', va='center',
                 fontsize=12,
                 color="black")

# Colorbar with label
cbar = plt.colorbar(img)
cbar.set_label("Cell Color Index (Each color represents a unique cell)")

# Optional class labels
classes = ["Class 0", "Class 1", "Class 2"]
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)

plt.tight_layout()
plt.show()

# ==========================================================
# 4️⃣ Bar Chart – Class-wise Accuracy
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

# Example confusion matrix (replace with your cm)
cm = np.array([[50,2,1],
               [4,45,3],
               [2,1,48]])

# Class names
classes = ["Class 0", "Class 1", "Class 2"]

# Calculate class-wise accuracy
class_accuracy = []
for i in range(cm.shape[0]):
    correct_predictions = cm[i, i]
    total_for_class = cm[i, :].sum()
    if total_for_class > 0:
        class_accuracy.append(correct_predictions / total_for_class)
    else:
        class_accuracy.append(0.0)

plt.figure(figsize=(8,5))

colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']

bars = plt.bar(
    range(len(class_accuracy)),
    class_accuracy,
    color=colors[:len(class_accuracy)],
    edgecolor='black',
    linewidth=1.5
)

plt.title("Class-wise Accuracy", fontsize=16, fontweight='bold')
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)

plt.ylim(0,1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add labels above each bar
for i, bar in enumerate(bars):
    height = bar.get_height()

    # Accuracy value
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.02,
        f"{height:.2f}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

    # Class name above value
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.08,
        classes[i],
        ha='center',
        fontsize=11,
        fontweight='bold',
        color='black'
    )

plt.tight_layout()
plt.show()
# ==========================================================
# 5️⃣ Pie Chart – Class Distribution
# ==========================================================
import matplotlib.pyplot as plt

support = cm.sum(axis=1)

plt.figure(figsize=(7,7))

# Modern colors
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6', '#F39C12']

explode = [0.05] * len(support)

wedges, texts, autotexts = plt.pie(
    support,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors[:len(support)],
    explode=explode,
    shadow=True,
    textprops={'fontsize':12, 'weight':'bold'}
)

plt.title("Class Distribution (%)",
          fontsize=16,
          fontweight='bold')

# ✅ Legend showing color meaning
plt.legend(
    wedges,
    [f"Class {i}" for i in range(len(support))],
    title="Classes",
    loc="best",
    fontsize=11
)

plt.axis('equal')
plt.tight_layout()
plt.show()

# ==========================================================
# 6️⃣ Histogram – Prediction Distribution
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))

# Example class names
classes = ["Class 0", "Class 1", "Class 2"]

colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(preds))))

# Create histogram
n, bins, patches = plt.hist(
    preds,
    bins=len(np.unique(preds)),
    edgecolor='black',
    linewidth=1.5
)

# Apply colors
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

# ⭐ Put numbers at center of each bar
for i in range(len(n)):
    center = (bins[i] + bins[i+1]) / 2
    plt.text(
        center,
        n[i] / 2,           # vertical center of bar
        int(n[i]),
        ha='center',
        va='center',
        fontsize=11,
        fontweight='bold',
        color='black'
    )

# Add class labels
plt.xticks(np.unique(preds), classes)

plt.title("Prediction Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Class")
plt.ylabel("Frequency")

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

import torch
import spacy
import gradio as gr
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
from sklearn.preprocessing import LabelEncoder

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
num_labels = 3  # adjust if needed

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
)

model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

# =========================
# Label Mapping (IMPORTANT)
# Must match training order
# =========================
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =========================
# Load spaCy for Aspect Extraction
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# Extract Aspects
# =========================
def extract_aspects(text):
    doc = nlp(text)
    aspects = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            aspects.append(token.text.lower())

    return list(set(aspects))

# =========================
# Predict Sentiment
# =========================
def predict_sentiment(text, aspect):
    encoding = tokenizer(
        aspect,
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    return label_map[pred.item()], float(confidence.item())

# =========================
# Full ABSA Pipeline
# =========================
def analyze_sentence(text):

    aspects = extract_aspects(text)

    if not aspects:
        return "No aspects detected."

    results = []

    for aspect in aspects:
        sentiment, confidence = predict_sentiment(text, aspect)
        results.append(
            f"Aspect: {aspect}\nSentiment: {sentiment}\nConfidence: {confidence:.2f}\n"
        )

    return "\n-----------------\n".join(results)

# =========================
# Gradio Interface
# =========================
interface = gr.Interface(
    fn=analyze_sentence,
    inputs=gr.Textbox(lines=3, placeholder="Enter sentence here..."),
    outputs=gr.Textbox(lines=20),
    title="MAMS ABSA Research Model",
    description="Enter a sentence. The model detects aspects and predicts sentiment."
)

interface.launch()
