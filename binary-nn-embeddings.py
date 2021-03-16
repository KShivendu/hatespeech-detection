import nltk
import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from utils import create_embedding_matrix
from utils import Attention
from utils import DataLoader
import pandas as pd
import zipfile
from utils import sensitivity, specificity, run_model_on_fold
from nn_models import build_model1, build_model3, build_model4, build_model5
pd.set_option('max_colwidth', 400)


X, y = DataLoader().get_data()
class_names = ['noHateSpeech', 'hateSpeech']
f = open("setC.txt", "a+")
f.write("{:<10} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(
    'Method', 'F1score', 'Precisi', 'Recall', 'Accurac', 'Specifi', 'Sensiti'))
f.write("=========================================================================\n")
f.close()
print("{:<10} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format(
    'Method', 'F1score', 'Precisi', 'Recall', 'Accurac', 'Specifi', 'Sensiti'))

# !wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
# !wget 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
# STORE THESE EMBEDDINGS IN './embeddings/crawl-300d-2M.vec' and './embeddings/glove.42B.300d.txt' respectively

embed_size = 300

n_fold = 10
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=7)

for emb_ma in [1, 2, 3]:
    print('*', end='')
    embed_size = 150  # * 2 = 300 for matrix 1 and 2
    if emb_ma == 3:
        embed_size = 300
    for max_len in [100, 150, 200, 250, 300]:
        print(".", end='')
        run_model_on_fold('b1_'+str(emb_ma)+'_'+str(max_len),
                          max_len, embed_size, emb_ma, build_model1)
        run_model_on_fold('b3_'+str(emb_ma)+'_'+str(max_len),
                          max_len, embed_size, emb_ma, build_model3)
        run_model_on_fold('b4_'+str(emb_ma)+'_'+str(max_len),
                          max_len, embed_size, emb_ma, build_model4)
        run_model_on_fold('b5_'+str(emb_ma)+'_'+str(max_len),
                          max_len, embed_size, emb_ma, build_model5)
