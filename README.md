# INLP Group Work — Options and Implementation Guide

Description of three simple project options that align with the GW notebook brief. Each option includes a short description and paste‑ready code for the `## Implementation` section of your notebook.

## Table of Contents
- Overview
- Requirements
- Options
  - Option 1: Emotion Detector
  - Option 2: Sentiment Analyzer
  - Option 3: Stance Classifier(s)
- Implementation
  - Reusable helpers
  - Per‑option snippets

## Overview
- Datasets: Uses Hugging Face `tweet_eval` subsets:
  - `emotion`, `hate`, `irony`, `offensive`, `sentiment`, `stance_abortion`, `stance_atheism`, `stance_climate`, `stance_feminist` (a.k.a. `stance_feminism`), `stance_hillary`.
- Approach per option:
  - Baseline: TF‑IDF + Logistic Regression (fast, interpretable)
  - Improved: DistilBERT fine‑tuning via `transformers` Trainer (compact, strong baseline)
  - Evaluation: Accuracy and macro‑F1, confusion matrix, brief error review

## Requirements
Install into the same environment as your Jupyter kernel:

```
python -m pip install -U datasets scikit-learn transformers torch ipywidgets
```

Classic Notebook widgets (only if using classic, not JupyterLab ≥3/VS Code):

```
jupyter nbextension enable --py widgetsnbextension
```

Widget rendering tips and troubleshooting: see `install_jupyter_notes.txt:1`.

## Options
- Option 1 — Emotion Detector
  - Classify short texts into emotions; include a minimal interactive demo.
  - Dataset: `emotion` (4 classes)
  - Pipeline: TF‑IDF + Logistic Regression, then DistilBERT fine‑tune
  - Metrics: Accuracy, macro‑F1, confusion matrix + brief error analysis

- Option 2 — Sentiment Analyzer
  - Classify text as negative/neutral/positive; optionally surface key phrases.
  - Dataset: `sentiment` (3 classes)
  - Pipeline: TF‑IDF + Linear SVM/LogReg, then DistilBERT fine‑tune
  - Metrics: Macro‑F1, class report, confusion matrix (optional calibration)

- Option 3 — Stance Classifier(s)
  - Predict stance (favor/against/neutral) on specific topics; compare topics.
  - Datasets: e.g., `stance_climate`, `stance_hillary` (and `stance_feminist`)
  - Pipeline: Reusable loop per dataset; TF‑IDF baseline + DistilBERT fine‑tune
  - Metrics: Per‑topic macro‑F1, confusion matrices; optional cross‑topic test

Across options, use this flow: setup/imports → data load/splits → baseline → transformer → error analysis → short discussion. If using widgets, ensure they render as noted above.

## Implementation

Below is a reusable implementation you can drop into a notebook. Then call the helpers with the subset you want for each option.

```
# --- Imports
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# --- Data loading helper (TweetEval subset)
def load_tweeteval(subset: str):
    """Load a TweetEval subset and return (ds, label_names).
    Common subsets: 'emotion', 'sentiment', 'stance_climate', 'stance_hillary', 'stance_feminist'.
    """
    try:
        ds = load_dataset('tweet_eval', subset)
    except Exception as e:
        # Some environments refer to the feminism subset as 'stance_feminism'
        if subset == 'stance_feminism':
            ds = load_dataset('tweet_eval', 'stance_feminist')
        else:
            raise e
    label_names = ds['train'].features['label'].names
    return ds, label_names

# --- Baseline: TF-IDF + Logistic Regression
def train_baseline(ds):
    X_train = ds['train']['text']
    y_train = ds['train']['label']
    X_val = ds['validation']['text']
    y_val = ds['validation']['label']

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)

    def eval_split(split):
        X = ds[split]['text']
        y = ds[split]['label']
        pred = pipe.predict(X)
        return {
            'acc': accuracy_score(y, pred),
            'macro_f1': f1_score(y, pred, average='macro'),
            'y_true': y,
            'y_pred': pred,
        }

    val_metrics = eval_split('validation')
    test_metrics = eval_split('test')
    return pipe, val_metrics, test_metrics

# --- Transformer fine-tune: DistilBERT
def finetune_transformer(ds, num_labels: int, model_name='distilbert-base-uncased', epochs=2):
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch['text'], truncation=True, padding=False)

    enc = ds.map(tokenize, batched=True)
    enc = enc.rename_column('label', 'labels')
    enc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir='./runs',
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_steps=50,
        learning_rate=2e-5,
        load_best_model_at_end=False,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'acc': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro'),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc['train'],
        eval_dataset=enc['validation'],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_val = trainer.evaluate(enc['validation'])
    eval_test = trainer.evaluate(enc['test'])
    return trainer, tok, eval_val, eval_test

# --- Simple error analysis helper
def show_errors(y_true, y_pred, texts, label_names, k=5):
    idx = np.where(np.array(y_true) != np.array(y_pred))[0][:k]
    for i in idx:
        print(f"Text: {texts[i]}\nTrue: {label_names[y_true[i]]} | Pred: {label_names[y_pred[i]]}\n---")
```

### Option 1 — Emotion Detector
- Subset: `emotion`
- Goal: Classify tweet emotions; provide a tiny demo cell to try your own text.

```
# Load data
ds, label_names = load_tweeteval('emotion')
print('Labels:', label_names)

# Baseline
baseline_model, val_b, test_b = train_baseline(ds)
print('Validation:', val_b)
print('Test:', test_b)
print(confusion_matrix(test_b['y_true'], test_b['y_pred']))
print(classification_report(test_b['y_true'], test_b['y_pred'], target_names=label_names))
show_errors(test_b['y_true'], test_b['y_pred'], ds['test']['text'], label_names)

# Transformer (2 quick epochs)
trainer, tok, val_t, test_t = finetune_transformer(ds, num_labels=len(label_names), epochs=2)
print('Val (transformer):', val_t)
print('Test (transformer):', test_t)

# Tiny interactive demo (classic notebook / VS Code)
from ipywidgets import Text, Button, Output, VBox
out = Output()
inp = Text(description='Text:', placeholder='Type a short text...')
btn = Button(description='Predict')

def on_click(_):
    with out:
        out.clear_output(wait=True)
        pred = baseline_model.predict([inp.value])[0]
        print('Predicted emotion:', label_names[pred])

btn.on_click(on_click)
VBox([inp, btn, out])
```

### Option 2 — Sentiment Analyzer
- Subset: `sentiment` (labels: negative/neutral/positive)

```
# Load data
ds, label_names = load_tweeteval('sentiment')
print('Labels:', label_names)

# Baseline
baseline_model, val_b, test_b = train_baseline(ds)
print('Validation:', val_b)
print('Test:', test_b)
print(confusion_matrix(test_b['y_true'], test_b['y_pred']))
print(classification_report(test_b['y_true'], test_b['y_pred'], target_names=label_names))

# Transformer
trainer, tok, val_t, test_t = finetune_transformer(ds, num_labels=len(label_names), epochs=2)
print('Val (transformer):', val_t)
print('Test (transformer):', test_t)
```

### Option 3 — Stance Classifier(s)
- Subsets: pick one or compare two, e.g., `stance_climate` and `stance_hillary`.
- Note: Some environments use `stance_feminist` (instead of `stance_feminism`). The loader handles both.

```
# Example A: stance on climate
ds_a, labels_a = load_tweeteval('stance_climate')
print('Labels A:', labels_a)
baseline_a, val_a, test_a = train_baseline(ds_a)
print('A Test:', test_a)
trainer_a, tok_a, val_ta, test_ta = finetune_transformer(ds_a, num_labels=len(labels_a), epochs=2)
print('A Test (transformer):', test_ta)

# Example B: stance on Hillary
ds_b, labels_b = load_tweeteval('stance_hillary')
print('Labels B:', labels_b)
baseline_b, val_b, test_b = train_baseline(ds_b)
print('B Test:', test_b)

# (Optional) Simple cross-topic test: train on A baseline, test on B
pred_cross = baseline_a.predict(ds_b['test']['text'])
print('Cross-topic baseline macro-F1:', f1_score(ds_b['test']['label'], pred_cross, average='macro'))
```

