import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
import nltk
from tqdm import tqdm

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Placeholder dataset class (user must provide clean data)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Data Augmentation Functions
def get_synonym(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().lower())
    return random.choice(synonyms) if synonyms else word

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    tfidf = TfidfVectorizer().fit_transform([sentence]).toarray()[0]
    word_scores = list(enumerate(tfidf))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    target_words = [words[i] for i, _ in word_scores[:max(1, int(0.2 * len(words)))]]
    
    non_target_indices = [i for i, word in enumerate(words) if word not in target_words]
    if len(non_target_indices) < n:
        return sentence
    replace_indices = random.sample(non_target_indices, n)
    
    new_words = words.copy()
    for idx in replace_indices:
        new_words[idx] = get_synonym(words[idx])
    return ' '.join(new_words)

def random_insertion(sentence, num=1):
    words = sentence.split()
    tfidf = TfidfVectorizer().fit_transform([sentence]).toarray()[0]
    word_scores = list(enumerate(tfidf))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    target_words = [words[i] for i, _ in word_scores[:max(1, int(0.2 * len(words)))]]
    
    non_target_words = [word for word in words if word not in target_words]
    if not non_target_words:
        return sentence
    insert_word = get_synonym(random.choice(non_target_words))
    
    insert_positions = random.sample(range(len(words) + 1), num)
    new_words = words.copy()
    for pos in sorted(insert_positions, reverse=True):
        new_words.insert(pos, insert_word)
    return ' '.join(new_words)

def augment_data(texts):
    augmented_texts = texts.copy()
    for text in texts:
        augmented_texts.append(synonym_replacement(text))
        augmented_texts.append(random_insertion(text))
    return augmented_texts

# Teacher Diversification
def fine_tune(model, dataloader, epochs=2, lr=0.1):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model

def re_init(model):
    for name, param in model.named_parameters():
        if 'bert.encoder.layer.7' in name or 'bert.encoder.layer.8' in name or \
           'bert.encoder.layer.9' in name or 'bert.encoder.layer.10' in name or \
           'bert.encoder.layer.11' in name:
            nn.init.xavier_uniform_(param.data)
    return model

def fine_pruning(model, dataloader, threshold=0.7):
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, 0, :]  # [CLS] token
            activations.append(hidden_states.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    mean_activations = np.mean(np.abs(activations), axis=0)
    
    prune_mask = mean_activations >= threshold
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.data[:, ~prune_mask] = 0
    return model

# Ensemble Distillation
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, student_hidden, teacher_hiddens, labels):
        ce_loss = self.ce_loss(student_outputs, labels)
        
        student_soft = torch.softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = torch.softmax(teacher_outputs / self.temperature, dim=1)
        kd_loss = self.kld_loss(torch.log_softmax(student_outputs / self.temperature, dim=1), teacher_soft) * (self.temperature ** 2)
        
        hd_loss = 0
        for s_hidden, t_hidden in zip(student_hidden, teacher_hiddens):
            s_norm = s_hidden / torch.norm(s_hidden, p=2, dim=-1, keepdim=True)
            t_norm = t_hidden / torch.norm(t_hidden, p=2, dim=-1, keepdim=True)
            hd_loss += torch.norm(s_norm - t_norm, p=2) ** 2
        hd_loss /= len(student_hidden)
        
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss + self.beta * hd_loss
        return total_loss

def ensemble_distillation(student, teachers, dataloader, epochs=3, lr=2e-5):
    optimizer = AdamW(student.parameters(), lr=lr)
    loss_fn = DistillationLoss()
    student.train()
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Distillation Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            student_outputs = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            student_logits = student_outputs.logits
            student_hidden = student_outputs.hidden_states[1:]  # Skip embedding layer
            
            teacher_logits = []
            teacher_hiddens = []
            for teacher in teachers:
                with torch.no_grad():
                    teacher.eval()
                    outputs = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    teacher_logits.append(outputs.logits)
                    teacher_hiddens.append(outputs.hidden_states[1:])
            
            avg_teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
            avg_teacher_hiddens = [torch.mean(torch.stack([h[i] for h in teacher_hiddens]), dim=0) for i in range(len(teacher_hiddens[0]))]
            
            loss = loss_fn(student_logits, avg_teacher_logits, student_hidden, avg_teacher_hiddens, labels)
            loss.backward()
            optimizer.step()
    return student

# Main BDEKD Pipeline
def bdekd_pipeline(texts, labels, model_name='bert-base-uncased', max_len=128):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = TextDataset(texts, labels, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Augment data
    augmented_texts = augment_data(texts)
    augmented_labels = labels * 3  # Tripled due to SR and RI
    aug_dataset = TextDataset(augmented_texts, augmented_labels, tokenizer, max_len)
    aug_dataloader = DataLoader(aug_dataset, batch_size=16, shuffle=True)
    
    # Initialize models
    backdoor_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    teacher1 = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    teacher2 = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    teacher3 = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    student = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    # Teacher Diversification
    teacher1 = fine_tune(teacher1, dataloader)
    teacher2 = re_init(teacher2)
    teacher2 = fine_tune(teacher2, dataloader)
    teacher3 = fine_pruning(teacher3, dataloader)
    teacher3 = fine_tune(teacher3, dataloader)
    
    # Ensemble Distillation
    teachers = [teacher1, teacher2, teacher3]
    student = ensemble_distillation(student, teachers, aug_dataloader)
    
    return student

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder for user-provided clean data
    texts = ["A sad, superior human comedy played out on the back roads of life.",
             "This movie is absolutely fantastic and inspiring."]
    labels = [0, 1]  # Negative, Positive
    
    student_model = bdekd_pipeline(texts, labels)
    print("BDEKD pipeline completed. Student model ready for evaluation.")
