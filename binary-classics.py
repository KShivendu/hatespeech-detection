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

# Run Naive Bayes

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
nested_cross_val(
    pipe, parameters, X, y, "MultiNB")
