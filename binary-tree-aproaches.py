from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import time
from utils import DataLoader, nested_cross_val


dl = DataLoader('data/Ethos_Dataset_Binary.csv')
X, y = dl.get_data()
f = open("res/setA.txt", "w+")
f.write("{: <7} | {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} {: <7} \n"
        .format('Method', 'Duration', 'scoreTi', 'F1', 'Prec.', 'Recall', 'Acc.', 'Spec.', 'Sens.'))
f.write("=========================================================================\n")
f.close()

# Run AdaBoost
ada = AdaBoostClassifier()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('ada', ada)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'ada__base_estimator':[None, DecisionTreeClassifier(max_depth=10), LogisticRegression(C=100)],
    'ada__n_estimators':[10, 50, 100, 300],
    'ada__learning_rate':[0.0001, 0.01, 0.5, 1]
}]
nested_cross_val(pipe, parameters, X, y, "AdaB",
                 n_jobs=22, filename='setB.txt')

# Run GradBoost
# grad = GradientBoostingClassifier()
# vec = TfidfVectorizer(analyzer='word')
# pipe = Pipeline(steps=[('vec', vec), ('grad', grad)])
# parameters = [{
#     'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
#     'vec__max_features':[5000, 10000, 50000, 100000],
#     'vec__stop_words':['english', None],
#     'grad__learning_rate':[0.0001, 0.01, 0.1, 0.5, 1],
#     'grad__n_estimators':[10, 50, 100, 300],
#     'grad__subsample':[0.7, 0.85, 1],
#     'grad__max_features':['sqrt', 'log2', None]
# }]
# nested_cross_val(pipe, parameters, X, y, "GradB")

# Run BaggingClsf
# bag = BaggingClassifier(n_jobs=-1)
# vec = TfidfVectorizer(analyzer='word')
# pipe = Pipeline(steps=[('vec', vec), ('bag', bag)])
# parameters = [{
#     'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
#     'vec__max_features':[5000, 10000, 50000, 100000],
#     'vec__stop_words':['english', None],
#     'bag__base_estimator':[None, DecisionTreeClassifier(max_depth=10), LogisticRegression(C=100)],
#     'bag__n_estimators':[10, 50, 100, 300],
#     'bag__max_samples':[0.7, 0.85, 1],
#     'bag__max_features':[0.5, 0.75, 1],
#     'bag__bootstrap':[True, False]
# }]
# nested_cross_val(pipe, parameters, X, y, "Bagging")
