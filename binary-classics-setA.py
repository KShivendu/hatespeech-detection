"""
In these experiments we will try logistic regression, svms, ridge, decision trees, naive bayes and random forests classifiers across a wide variety of parameters for each algorithm and test them via nested cross validation method.
"""

from utils import DataLoader, nested_cross_val
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

dl = DataLoader('data/Ethos_Dataset_Binary.csv')
X, y = dl.get_data()
f = open("res/setA.txt", "w+")
f.write("{: <7} | {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} \n"
        .format('Method', 'Duration', 'scoreTi', 'F1', 'Prec.', 'Recall', 'Acc.', 'Spec.', 'Sens.'))
f.write("=========================================================================\n")
f.close()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Run Naive Bayes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
mNB = MultinomialNB()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(
    steps=[('vec', vec), ('mNB', mNB)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'mNB__alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
}]
nested_cross_val(pipe, parameters, X, y, "MultiNB",
                 n_jobs=18, filename='setA.txt')

bNB = BernoulliNB(binarize=0.5)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('bNB', bNB)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'bNB__alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
}]
nested_cross_val(pipe, parameters, X, y, "BernouNB")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Run Logistic Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('log', log)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'log__C':[0.5, 1, 3, 5, 10, 1000],
    'log__solver':['newton-cg', 'lbfgs', 'sag'],
    'log__penalty':['l2']
}, {
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'log__C':[0.5, 1, 3, 5, 10, 1000],
    'log__solver':['saga'],
    'log__penalty':['l1']
}]
nested_cross_val(pipe, parameters, X, y, "LogReg")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Run SVM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
svm = SVC(random_state=0)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('svm', svm)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'svm__kernel':['rbf'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000],
    'svm__gamma':[0.05, 0.1, 0.5, 0.9, 1]
}, {
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'svm__kernel':['linear'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000]
}]
nested_cross_val(pipe, parameters, X, y, "SVM")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Run RidgeClassifier
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ridge = RidgeClassifier(random_state=0, fit_intercept=False)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('ridge', ridge)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'ridge__solver':['cholesky', 'lsqr', 'sparse_cg', 'saga'],
    'ridge__alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0]
}]
nested_cross_val(pipe, parameters, X, y, "Ridge")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Run DecisionTree
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dTree = DecisionTreeClassifier(random_state=0)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('dTree', dTree)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'dTree__criterion':['gini', 'entropy'],
    'dTree__max_depth':[1, 2, 3, 4, 5, 10, 25, 50, 100, 200],
    'dTree__max_features':[2, 3, 4, 5, 'sqrt', 'log2', None],
    'dTree__min_samples_leaf': [1, 2, 3, 4, 5],
    'dTree__min_samples_split': [2, 4, 8, 10, 12]
}]
nested_cross_val(pipe, parameters, X, y, "DTree")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            Run RandomForest
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
randFor = RandomForestClassifier(random_state=0, n_jobs=-1)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('randFor', randFor)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'randFor__max_depth':[1, 10, 50, 100, 200],
    'randFor__max_features':['sqrt', 'log2', None],
    'randFor__bootstrap':[True, False],
    'randFor__n_estimators': [10, 100, 500, 1000]
}]
nested_cross_val(pipe, parameters, X, y, "RandomForest")
