# pip install transformers

from utils import specificity, sensitivity
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from utilities.preprocess import Preproccesor
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import time
import numpy as np
import nltk
import tensorflow as tf
import transformers

model_class = transformers.BertModel
tokenizer_class = transformers.BertTokenizer
pretrained_weights = 'bert-base-uncased'
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)


max_seq = 100


def tokenize_text(df, max_seq):
    return [
        tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.comment_text.values
    ]


def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])


def tokenize_and_pad_text(df, max_seq):
    tokenized_text = tokenize_text(df, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)


def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)


# ORIGINAL CODE
pd.set_option('max_colwidth', 400)
nltk.download('wordnet')
nltk.download('stopwords')
X, y = Preproccesor.load_data(True)
class_names = ['noHateSpeech', 'hateSpeech']

f_results = []
n_fold = 10
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=7)
counter = 0
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_test = X[train_index], X[valid_index]
    y_train, y_test = y[train_index], y[valid_index]
    ids = []
    idsT = []
    for i in range(0, len(X_train)):
        ids.append(i)
    for i in range(0, len(X_test)):
        idsT.append(i)
    train = pd.DataFrame(X_train, columns=['comment_text'], index=ids)
    train['target'] = y_train
    test = pd.DataFrame(X_test, columns=['comment_text'], index=idsT)
    test['target'] = y_test
    train.to_csv('train.tsv', sep='\t', index=False, header=False)
    test.to_csv('test.tsv', sep='\t', index=False, header=True)
    myparam = {
        "DATA_COLUMN": "comment_text",
        "LABEL_COLUMN": "target",
        "MAX_SEQ_LENGTH": 100,
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS": 10,
        "SAVE_SUMMARY_STEPS": 100,  # 100
        "SAVE_CHECKPOINTS_STEPS": 10000  # 10000
    }
    result, estimator = run_on_dfs(train, test, **myparam)
    f_results.append(result)
    print(len(f_results), result)
    del result, estimator
    # !rm '/content/test.tsv'
    # !rm '/content/train.tsv'
    # !rm - R '/content/output'


# END ORIGINAL CODE
train_indices = tokenize_and_pad_text(df_train, max_seq)
val_indices = tokenize_and_pad_text(df_val, max_seq)
test_indices = tokenize_and_pad_text(df_test, max_seq)
with torch.no_grad():
    x_train = bert_model(train_indices)[0]
    x_val = bert_model(val_indices)[0]
    x_test = bert_model(test_indices)[0]
y_train = targets_to_tensor(df_train, target_columns)
y_val = targets_to_tensor(df_val, target_columns)
y_test = targets_to_tensor(df_test, target_columns)
