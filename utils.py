# NN utils
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.engine import Layer

from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# for nested_cross_val

from sklearn.metrics import make_scorer, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
import numpy as np

# Preprocessing
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

import time

import nltk
nltk.download('wordnet')
nltk.download('stopwords')


class DataLoader:
    def __init__(cls, filename: str = "../ethos/ethos_data/Ethos_Dataset_Binary.csv", lang='en', cleaner=None) -> None:
        cls.filename = filename
        cls.lang = lang
        cls.lemmatizer = WordNetLemmatizer()
        cls.eng_stemmer = SnowballStemmer("english", ignore_stopwords=True)
        cls.stopwords = stopwords.words('english')
        cls.cleaner = cleaner
        class_names = ['not-hate-speech', 'hate-speech']

    def get_data(cls, preprocessed=True, stemming=True):
        data = pd.read_csv(cls.filename, delimiter=';')
        print(f"Loaded file : {cls.filename.split('/')[-1]}")
        np.random.seed(2000)
        data = data.iloc[np.random.permutation(len(data))]
        XT, yT = data['comment'].values, data['isHate'].values
        X, y = [], []
        for yt in yT:
            if yt >= 0.5:
                y.append(int(1))
            else:
                y.append(int(0))
        for x in XT:
            if preprocessed:
                if cls.cleaner:
                    X.append(cls.cleaner(str(x)))
                else:
                    X.append(cls._cleaner(str(x), False, stemming))
            else:
                X.append(x)
        return np.array(X), np.array(y)

    def _cleaner(cls, text, stops=False, stemming=False):
        text = str(text)
        text = re.sub(r" US ", " american ", text)
        text = text.lower().split()
        text = " ".join(text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"don't", "do not ", text)
        text = re.sub(r"aren't", "are not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"%", " percent ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"doesn't", "does not ", text)
        text = re.sub(r"he's", "he is ", text)
        text = re.sub(r"she's", "she is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower().split()
        text = [w for w in text if len(w) >= 2]
        if stops:
            text = [word for word in text if word not in cls.stopwords]
        if stemming:
            text = [cls.eng_stemmer.stem(word) for word in text]
            text = [cls.lemmatizer.lemmatize(word) for word in text]
        text = " ".join(text)
        return text


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def get_config(self):  # NEWLY ADDED SO WILL HAVE TO CHECK
        config = super().get_config().copy()
        config.update({

            'supports_masking': self.supports_masking,
            'init': self.init,

            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,

            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,

            'bias': self.bias,
            'step_dim': self.step_dim,
            'features_dim': self.features_dim
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tn+fp) > 0:
        speci = tn/(tn+fp)
        return speci
    return 0


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tp+fn) > 0:
        sensi = tp/(tp+fn)
        return sensi
    return 0


def print_save(text, path, method='a+'):
    print(text)
    f = open(path, method)
    f.write(text)
    f.close()


def merge_dicts(dict_elements, dict_with_lists):
    for k, v in dict_elements.items():
        dict_with_lists[k].append(v)

    return dict_with_lists


def nested_cross_val(pipe, parameters, X, y, name, n_jobs=18, filename='setA.txt'):
    scores = {
        'fit_time': [],
        'score_time': [],
        'test_F1': [],
        'test_Precision': [],
        'test_Recall': [],
        'test_Accuracy': [],
        'test_Specificity': [],
        'test_Sensitivity': [],
    }

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)
    splits = outer_cv.split(X)
    for train_index, test_index in splits:
        X_trainO, X_testO = X[train_index], X[test_index]
        y_trainO, y_testO = y[train_index], y[test_index]
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
        clf = GridSearchCV(estimator=pipe, param_grid=parameters,
                           cv=inner_cv, n_jobs=n_jobs, verbose=1, scoring='f1')
        a = time.time()
        clf.fit(X_trainO, y_trainO)
        fit_time = time.time() - a
        a = time.time()
        y_preds = clf.predict(X_testO)
        score_time = time.time() - a
        current_scores = {
            'fit_time': fit_time,
            'score_time': score_time,
            'test_F1': f1_score(y_testO, y_preds, average='macro'),
            'test_Precision': precision_score(y_testO, y_preds, average='macro'),
            'test_Recall': recall_score(y_testO, y_preds, average='macro'),
            'test_Accuracy': accuracy_score(y_testO, y_preds),
            'test_Specificity': specificity(y_testO, y_preds),
            'test_Sensitivity': sensitivity(y_testO, y_preds),
        }
        scores = merge_dicts(current_scores, scores)

    print(scores)

    for k in scores:
        print(f"{name} {k}: {sum(scores[k])/10}")
    print_save(
        "{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(
            str(name)[:7],
            str('%.4f' % (sum(scores['fit_time'])/10)),
            str('%.4f' % (sum(scores['score_time'])/10)),
            str('%.4f' % (sum(scores['test_F1'])/10)),
            str('%.4f' % (sum(scores['test_Precision'])/10)),
            str('%.4f' % (sum(scores['test_Recall'])/10)),
            str('%.4f' % (sum(scores['test_Accuracy'])/10)),
            str('%.4f' % (sum(scores['test_Specificity'])/10)),
            str('%.4f' % (sum(scores['test_Sensitivity'])/10))),
        f'res/{filename}',
    )


def build_matrix(embedding_path, tk, max_features):
    embedding_index = dict(get_coefs(*o.strip().split(" "))
                           for o in open(embedding_path, encoding="utf-8"))

    word_index = tk.word_index
    nb_words = max_features
    embedding_matrix = np.zeros((nb_words + 1, 300))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_embedding_matrix(embed, tk, max_features, fasttext='./embeddings/crawl-300d-2M.vec', glove='./embeddings/glove.42B.300d.txt'):
    if embed == 1:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        return build_matrix(fasttext, tk, max_features)
    elif embed == 2:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print('http://nlp.stanford.edu/data/glove.42B.300d.zip')
        return build_matrix(glove, tk, max_features)
    else:
        print("Please download and put this embeddings to the folder ethos/experiments/embeddings: ")
        print(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        print('http://nlp.stanford.edu/data/glove.42B.300d.zip')
        return np.concatenate([build_matrix(fasttext, tk, max_features), build_matrix(glove, tk, max_features)], axis=-1)


def run_model_on_fold(name, max_len, embed_size, embed, bulid_fun, folds,X,y):
    max_features = 50000
    scores = {
        'fit_time': [],
        'score_time': [],
        'test_F1': [],
        'test_Precision': [],
        'test_Recall': [],
        'test_Accuracy': [],
        'test_Specificity': [],
        'test_Sensitivity': [],
    }
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        tk = Tokenizer(lower=True, filters='',
                       num_words=max_features, oov_token=True)
        tk.fit_on_texts(X_train)
        train_tokenized = tk.texts_to_sequences(X_train)
        valid_tokenized = tk.texts_to_sequences(X_valid)
        X_train = pad_sequences(train_tokenized, maxlen=max_len)
        X_valid = pad_sequences(valid_tokenized, maxlen=max_len)
        embedding_matrix = create_embedding_matrix(embed, tk, max_features)

        model = bulid_fun(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix,
                          lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4,
                          fold_id=fold_n)

        y_preds = []
        for i in model.predict(X_valid):
            if i[0] >= 0.5:
                y_preds.append(1)
            else:
                y_preds.append(0)
        print(accuracy_score(y_valid, y_preds))
        scores['test_F1'].append(f1_score(y_valid, y_preds, average='macro'))
        scores['test_Precision'].append(
            precision_score(y_valid, y_preds, average='macro'))
        scores['test_Recall'].append(
            recall_score(y_valid, y_preds, average='macro'))
        scores['test_Accuracy'].append(accuracy_score(y_valid, y_preds))
        scores['test_Specificity'].append(specificity(y_valid, y_preds))
        scores['test_Sensitivity'].append(sensitivity(y_valid, y_preds))
    print("{:<10} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format(str(name)[:7],
                                                                str('%.4f' % (
                                                                    sum(scores['test_F1']) / 10)),
                                                                str('%.4f' % (
                                                                    sum(scores['test_Precision']) / 10)),
                                                                str('%.4f' % (
                                                                    sum(scores['test_Recall']) / 10)),
                                                                str('%.4f' % (
                                                                    sum(scores['test_Accuracy']) / 10)),
                                                                str('%.4f' % (
                                                                    sum(scores['test_Specificity']) / 10)),
                                                                str('%.4f' % (sum(scores['test_Sensitivity']) / 10))))
    f = open("setC.txt", "a+")
    f.write("{:<10} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format(str(name)[:7],
                                                                  str('%.4f' % (
                                                                      sum(scores['test_F1']) / 10)),
                                                                  str('%.4f' % (
                                                                      sum(scores['test_Precision']) / 10)),
                                                                  str('%.4f' % (
                                                                      sum(scores['test_Recall']) / 10)),
                                                                  str('%.4f' % (
                                                                      sum(scores['test_Accuracy']) / 10)),
                                                                  str('%.4f' % (
                                                                      sum(scores['test_Specificity']) / 10)),
                                                                  str('%.4f' % (sum(scores['test_Sensitivity']) / 10)))+'\n')
    f.close()
