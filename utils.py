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


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tn+fp) > 0:
        speci = tn/(tn+fp)
        return speci
    return 0


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
