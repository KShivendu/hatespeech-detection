from .utils import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


dl = DataLoader('data/Ethos_Dataset_Binary.csv')
X, y = dl.get_data()
print(len(X), len(y))
print(f'{len(y)-sum(y)} (label : 0) + {sum(y)} (label : 1) = {len(y)}')
print(X[0])
print(y[0])
class_names = ['not-hate-speech', 'hate-speech']

# ML Model
kf = KFold(n_splits=10)
kf.get_n_splits()

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 5), max_features=50000)
    vec.fit(X_train)
    X_tr = vec.transform(X_train)
    X_te = vec.transform(X_test)
    X_tw = vec.transform(X)
    svm = SVC(kernel='rbf')
    svm.fit(X_tr, y_train)

    y_predict = svm.predict(X_te)
    print('F1 : ', f1_score(y_test, y_predict, average='weighted'))
    print('CF-Mat : \n', confusion_matrix(y_test, y_predict))
