#@title K-Tran ABSA: Complete Training and Testing (Direct File Upload)
# 1. SETUP: Install all necessary libraries
# ==============================================================================
!pip install transformers torch scikit-learn lxml spacy gradio
!python -m spacy download en_core_web_sm

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from tqdm.notebook import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import spacy
from lxml import etree
import gradio as gr
from google.colab import files
import shutil

# ==============================================================================
# 2. UPLOAD AND PREPARE DATA (Handles individual file uploads)
# ==============================================================================
# Create the target directory
data_dir = '/content/semeval14'
os.makedirs(data_dir, exist_ok=True)

print("Please upload 'Restaurants_Train.xml' and 'Restaurants_Test_Gold.xml'.")
uploaded = files.upload()

# Verify and move the uploaded files to the correct directory
required_files = ['Restaurants_Train.xml', 'Restaurants_Test_Gold.xml']
for req_fn in required_files:
    found_uploaded_fn = None
    # Extract base name for matching, e.g., 'Restaurants_Train'
    req_base_name = os.path.splitext(req_fn)[0]

    for uploaded_fn in uploaded.keys():
        # Check if the uploaded filename starts with the required base name
        # and ends with '.xml' (to allow for Colab's (n) suffix)
        if uploaded_fn.startswith(req_base_name) and uploaded_fn.endswith('.xml'):
            found_uploaded_fn = uploaded_fn
            break

    if found_uploaded_fn:
        # Move the uploaded file (e.g., 'Restaurants_Train (1).xml')
        # to the data_dir with the required name (e.g., 'Restaurants_Train.xml')
        source_path = os.path.join('/content', found_uploaded_fn)
        destination_path = os.path.join(data_dir, req_fn)
        shutil.move(source_path, destination_path)
        print(f'Successfully prepared "{req_fn}" from uploaded file "{found_uploaded_fn}"')
    else:
        print(f'Error: Required file "{req_fn}" was not uploaded or matched.')

# ==============================================================================
# 3. CONFIGURATION
# ==============================================================================
config = {
    'data': {
        'train_path': '/content/semeval14/Restaurants_Train.xml',
        'test_path': '/content/semeval14/Restaurants_Test_Gold.xml',
        'processed_dir': '/content/data/processed',
        'max_seq_len': 128,
        'sentiment_map': {'positive': 2, 'neutral': 1, 'negative': 0}
    },
    'model': {
        'base_model': 'roberta-base',
        'k_tran_layers': 3,
        'k_tran_heads': 8,
        'dropout_rate': 0.1
    },
    'training': {
        'optimizer': 'adamw',
        'learning_rate': 2e-5,
        'epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 2,
        'num_workers': 2,
        'ate_loss_weight': 0.2,
        'max_grad_norm': 1.0
    },
    'environment': {
        'device': 'cuda',
        'seed': 42
    }
}

# ==============================================================================
# 4. MODEL DEFINITION (K-Tran)
# ==============================================================================
class AspectAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AspectAwareAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multihead_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class KTranEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(KTranEncoderLayer, self).__init__()
        self.self_attn = AspectAwareAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, syntax_matrix=None):
        key_padding_mask = (src_mask == 0) if src_mask is not None else None

        attn_mask = None
        if syntax_matrix is not None:
            num_heads = self.self_attn.multihead_attn.num_heads
            attn_mask = syntax_matrix.repeat_interleave(num_heads, dim=0)

        attn_output, _ = self.self_attn(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

class KTranForABSA(nn.Module):
    def __init__(self, model_config):
        super(KTranForABSA, self).__init__()
        self.config = model_config
        self.roberta = RobertaModel.from_pretrained(self.config['base_model'])
        d_model = self.roberta.config.hidden_size
        nhead = self.config['k_tran_heads']
        num_layers = self.config['k_tran_layers']
        self.k_tran_encoder = nn.ModuleList(
            [KTranEncoderLayer(d_model, nhead) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        self.sentiment_num_labels = 3
        self.sentiment_classifier = nn.Linear(d_model, self.sentiment_num_labels)
        self.ate_num_labels = 3
        self.ate_classifier = nn.Linear(d_model, self.ate_num_labels)

    def forward(self, input_ids, attention_mask, syntax_matrix=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_outputs.last_hidden_state

        syntax_bias = None
        if syntax_matrix is not None:
            syntax_bias = (1.0 - syntax_matrix) * -1e9

        encoder_output = sequence_output
        for layer in self.k_tran_encoder:
            encoder_output = layer(encoder_output, src_mask=attention_mask, syntax_matrix=syntax_bias)

        encoder_output = self.dropout(encoder_output)
        cls_token_representation = encoder_output[:, 0]
        sentiment_logits = self.sentiment_classifier(cls_token_representation)
        ate_logits = self.ate_classifier(encoder_output)
        return {"sentiment_logits": sentiment_logits, "ate_logits": ate_logits}

# ==============================================================================
# 5. DATA PROCESSING
# ==============================================================================
nlp = spacy.load("en_core_web_sm")

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.config['model']['base_model'])
        self.max_seq_len = self.config['data']['max_seq_len']
        self.sentiment_map = self.config['data']['sentiment_map']

    def _parse_xml_lxml(self, file_path):
        try:
            tree = etree.parse(file_path)
            root = tree.getroot()
        except (etree.XMLSyntaxError, IOError) as e:
            print(f"Error parsing XML file {file_path}: {e}")
            return []
        all_data = []
        sentences = root.xpath('//sentence')
        for sentence_node in sentences:
            text = ''.join(sentence_node.xpath('./text/text()'))
            if not text: continue
            aspects_data = []
            for aspect_term_node in sentence_node.xpath('.//aspectTerm'):
                term = aspect_term_node.get('term')
                polarity = aspect_term_node.get('polarity')
                if term and polarity and polarity in self.sentiment_map:
                    aspects_data.append({'term': term, 'polarity': polarity})
            if text and aspects_data:
                all_data.append({'text': text, 'aspects': aspects_data})
        return all_data

    def _create_syntax_matrix(self, text, encoded):
        doc = nlp(text)
        seq_len = encoded['input_ids'].shape[1]
        matrix = torch.zeros(seq_len, seq_len)
        offset_mapping = encoded.offset_mapping[0]
        for token in doc:
            char_start, char_end = token.idx, token.idx + len(token.text)
            try:
                token_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < char_end and end > char_start]
                if not token_indices: continue
                head_char_start, head_char_end = token.head.idx, token.head.idx + len(token.head.text)
                head_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < head_char_end and end > head_char_start]
                if head_indices:
                    for i in token_indices:
                        for j in head_indices:
                            matrix[i][j] = 1
                            matrix[j][i] = 1
            except IndexError:
                continue
        for i in range(seq_len):
            matrix[i][i] = 1
        return matrix.float()

    def _tokenize_and_prepare(self, text, aspect_term):
        encoded = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        offset_mapping = encoded['offset_mapping'].squeeze(0)
        syntax_matrix = self._create_syntax_matrix(text, encoded)
        ate_labels = torch.zeros(self.max_seq_len, dtype=torch.long)
        try:
            aspect_start_char = text.find(aspect_term)
            if aspect_start_char != -1:
                aspect_end_char = aspect_start_char + len(aspect_term)
                token_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < aspect_end_char and end > aspect_start_char]
                if token_indices:
                    ate_labels[token_indices[0]] = 1 # B-Aspect
                    for i in token_indices[1:]:
                        ate_labels[i] = 2 # I-Aspect
        except Exception:
            pass
        return input_ids, attention_mask, ate_labels, syntax_matrix

    def process(self, file_path, name="train"):
        raw_data = self._parse_xml_lxml(file_path)
        processed_data = []
        for item in tqdm(raw_data, desc=f"Processing {name} data"):
            text = item['text']
            for aspect in item['aspects']:
                aspect_term, polarity = aspect['term'], aspect['polarity']
                input_ids, attention_mask, ate_labels, syntax_matrix = self._tokenize_and_prepare(text, aspect_term)
                sentiment_label = self.sentiment_map[polarity]
                processed_data.append({
                    'input_ids': input_ids, 'attention_mask': attention_mask,
                    'ate_labels': ate_labels, 'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
                    'syntax_matrix': syntax_matrix
                })
        output_dir = self.config['data']['processed_dir']
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}_data.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Saved processed {name} data to {output_path}. Total samples: {len(processed_data)}")

# ==============================================================================
# 6. TRAINING
# ==============================================================================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['environment']['device'] if torch.cuda.is_available() else 'cpu')
        self.model = KTranForABSA(self.config['model']).to(self.device)
        self.epochs = self.config['training']['epochs']
        self.train_batch_size = self.config['training']['train_batch_size']
        self.eval_batch_size = self.config['training']['eval_batch_size']
        self.learning_rate = float(self.config['training']['learning_rate'])
        self.ate_loss_weight = self.config['training']['ate_loss_weight']
        self.gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.num_workers = self.config['training']['num_workers']

    def _load_data(self, name="train"):
        data_path = os.path.join(self.config['data']['processed_dir'], f"{name}_data.pkl")
        with open(data_path, 'rb') as f: data = pickle.load(f)
        dataset = TensorDataset(
            torch.stack([item['input_ids'] for item in data]),
            torch.stack([item['attention_mask'] for item in data]),
            torch.stack([item['ate_labels'] for item in data]),
            torch.stack([item['sentiment_label'] for item in data]),
            torch.stack([item['syntax_matrix'] for item in data])
        )
        batch_size = self.train_batch_size if name == "train" else self.eval_batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=(name=="train"), num_workers=self.num_workers, pin_memory=True)

    def _compute_metrics(self, sentiment_preds, sentiment_labels, ate_preds, ate_labels, attention_mask):
        sentiment_preds = np.argmax(sentiment_preds, axis=1).flatten()
        sentiment_labels = sentiment_labels.flatten()
        acc = accuracy_score(sentiment_labels, sentiment_preds)
        f1 = f1_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0)
        prec = precision_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0)
        rec = recall_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0)
        active_loss = attention_mask.view(-1) == 1
        active_logits = ate_preds.view(-1, self.model.ate_num_labels)
        active_labels = ate_labels.view(-1)
        active_preds = torch.argmax(active_logits, axis=1)
        ate_f1 = f1_score(active_labels[active_loss].cpu(), active_preds[active_loss].cpu(), average='weighted', zero_division=0)
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "ate_f1": ate_f1}

    def train(self):
        train_dataloader = self._load_data("train")
        test_dataloader = self._load_data("test")
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        sentiment_loss_fn = nn.CrossEntropyLoss()
        ate_loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler()
        best_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            optimizer.zero_grad()
            for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
                input_ids, attention_mask, ate_labels, sentiment_labels, syntax_matrix = batch
                with autocast():
                    outputs = self.model(input_ids, attention_mask, syntax_matrix=syntax_matrix)
                    loss_sentiment = sentiment_loss_fn(outputs['sentiment_logits'], sentiment_labels)
                    active_loss_mask = attention_mask.view(-1) == 1
                    active_logits = outputs['ate_logits'].view(-1, self.model.ate_num_labels)
                    active_labels = ate_labels.view(-1)
                    loss_ate = ate_loss_fn(active_logits[active_loss_mask], active_labels[active_loss_mask])
                    loss = (loss_sentiment + self.ate_loss_weight * loss_ate) / self.gradient_accumulation_steps
                scaler.scale(loss).backward()
                total_loss += loss.item()
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            avg_train_loss = total_loss / len(train_dataloader) * self.gradient_accumulation_steps
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            current_f1 = self.evaluate(test_dataloader)
            if current_f1 > best_f1:
                best_f1 = current_f1
                print("New best model found! Saving...")
                os.makedirs('/content/models', exist_ok=True)
                torch.save(self.model.state_dict(), '/content/models/k_tran_best.pt')

    def evaluate(self, dataloader):
        self.model.eval()
        all_sentiment_preds, all_sentiment_labels, all_ate_preds, all_ate_labels, all_masks = [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
                input_ids, attention_mask, ate_labels, sentiment_labels, syntax_matrix = batch
                with autocast():
                    outputs = self.model(input_ids, attention_mask, syntax_matrix=syntax_matrix)
                all_sentiment_preds.append(outputs['sentiment_logits'].cpu().numpy())
                all_sentiment_labels.append(sentiment_labels.cpu().numpy())
                all_ate_preds.append(outputs['ate_logits'].cpu().numpy())
                all_ate_labels.append(ate_labels.cpu().numpy())
                all_masks.append(attention_mask.cpu().numpy())
        metrics = self._compute_metrics(
            torch.from_numpy(np.concatenate(all_sentiment_preds, axis=0)),
            torch.from_numpy(np.concatenate(all_sentiment_labels, axis=0)),
            torch.from_numpy(np.concatenate(all_ate_preds, axis=0)),
            torch.from_numpy(np.concatenate(all_ate_labels, axis=0)),
            torch.from_numpy(np.concatenate(all_masks, axis=0))
        )
        print(f"Evaluation Results: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ATE F1: {metrics['ate_f1']:.4f}")
        return metrics['f1']

# ==============================================================================
# 7. RUN DATA PROCESSING AND TRAINING
# ==============================================================================
print("--- Starting Data Processing ---")
data_processor = DataProcessor(config)
data_processor.process(config['data']['train_path'], "train")
data_processor.process(config['data']['test_path'], "test")

print("\n--- Starting Model Training ---")
trainer = Trainer(config)
trainer.train()
print("\n--- Training Complete ---")

# ==============================================================================
# 8. SETUP INTERACTIVE DEMO
# ==============================================================================
print("\n--- Loading final model for interactive demo ---")
device = torch.device(config['environment']['device'] if torch.cuda.is_available() else 'cpu')
final_model = KTranForABSA(config['model'])
final_model.load_state_dict(torch.load('/content/models/k_tran_best.pt', map_location=device))
final_model.to(device)
final_model.eval()
tokenizer = RobertaTokenizerFast.from_pretrained(config['model']['base_model'])
nlp_demo = spacy.load("en_core_web_sm")

def predict_sentiment(sentence):
    max_seq_len = config['data']['max_seq_len']
    encoded = tokenizer.encode_plus(
        sentence, add_special_tokens=True, max_length=max_seq_len,
        padding='max_length', truncation=True, return_tensors='pt',
        return_offsets_mapping=True
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    doc = nlp_demo(sentence)
    matrix = torch.zeros(max_seq_len, max_seq_len, dtype=torch.float32)
    offset_mapping = encoded.offset_mapping.squeeze(0).tolist()
    for token in doc:
        char_start, char_end = token.idx, token.idx + len(token.text)
        token_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < char_end and end > char_start]
        if not token_indices: continue
        head_char_start, head_char_end = token.head.idx, token.head.idx + len(token.head.text)
        head_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < head_char_end and end > head_char_start]
        if head_indices:
            for i in token_indices:
                for j in head_indices:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
    for i in range(max_seq_len): matrix[i][i] = 1
    syntax_matrix = matrix.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = final_model(input_ids, attention_mask, syntax_matrix=syntax_matrix)

    ate_preds = torch.argmax(outputs['ate_logits'], dim=2).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    aspects = []
    current_aspect_tokens = []
    for i, pred in enumerate(ate_preds):
        if not attention_mask[0, i] or tokens[i] in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue

        token_id = tokenizer.convert_tokens_to_ids(tokens[i])
        if pred == 1: # B-Aspect
            if current_aspect_tokens: aspects.append(tokenizer.decode(current_aspect_tokens))
            current_aspect_tokens = [token_id]
        elif pred == 2 and current_aspect_tokens: # I-Aspect
            current_aspect_tokens.append(token_id)
        else: # O-token
            if current_aspect_tokens:
                aspects.append(tokenizer.decode(current_aspect_tokens))
                current_aspect_tokens = []
    if current_aspect_tokens: aspects.append(tokenizer.decode(current_aspect_tokens))

    sentiment_pred = torch.argmax(outputs['sentiment_logits'], dim=1).item()
    sentiment_map_inv = {v: k for k, v in config['data']['sentiment_map'].items()}
    sentiment = sentiment_map_inv.get(sentiment_pred, "unknown")

    if not aspects:
        return "No aspects detected.", {}

    results = {}
    for aspect in aspects:
        results[aspect.strip()] = sentiment.upper()

    return f"Overall Sentiment: {sentiment.upper()}", results

# Launch Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Enter a sentence", placeholder="The food was amazing but the service was slow."),
    outputs=[gr.Textbox(label="Overall Sentiment"), gr.Label(label="Aspects and Sentiments")],
    title="🧪 K-Tran ABSA: Interactive Demo",
    description="Test the K-Tran Syntax-Aware Model. This model identifies aspects in a sentence and classifies their sentiment.",
    examples=[
        ["The service is excellent but the food is terrible."],
        ["I loved the ambiance, and the pasta was cooked perfectly."],
        ["The sushi was incredibly fresh and the presentation was beautiful."]
    ]
)
interface.launch(debug=True)