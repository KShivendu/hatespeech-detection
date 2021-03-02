import pandas as pd
import numpy as np

# Preprocessing
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


class DataLoader:
    def __init__(cls, filename: str = "../ethos/ethos_data/Ethos_Dataset_Binary.csv", cleaner=None) -> None:
        cls.filename = filename
        cls.lemmatizer = WordNetLemmatizer()
        cls.eng_stemmer = SnowballStemmer("english", ignore_stopwords=True)
        cls.stopwords = stopwords.words('english')
        cls.cleaner = cleaner

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
