import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier

###############################  Classifiers  ##########################
names = ["LDA", "SVM", "DT-Gini", "DT-ig", "RandomForest"]
classifiers = [LinearDiscriminantAnalysis(),
               svm.LinearSVC(C=1, random_state=0),
               DecisionTreeClassifier(criterion="gini", max_leaf_nodes=20),
               DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=20),
               RandomForestClassifier(criterion="gini", max_depth=3, random_state=0)]

###############################  DATASETS  ##########################
datasets = np.genfromtxt("spambase.data", delimiter=",")
observation, features = datasets.shape

X = []; y = []
i = 1; j = 0
scoring = ['f1_macro']

file1 = open("spambase.data", 'r')
for data in file1:
    rc = data.split(',')
    X.append(rc[:-1])
    y.append(rc[features - 1].strip('\n'))

###########################  TRAIN AND PLOT  #########################
linearly_separable = (X, y)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=5)

for clf in classifiers:

    clf.fit(X_train, y_train)
    F_msr = cross_validate(clf, X, y, scoring=scoring, cv=10, return_train_score=False)
    y_pred = clf.predict(X_test)
    y_pred, y_test = list(map(int, y_pred)), list(map(int, y_test))
    # matrix = metrics.confusion_matrix(y_test, y_pred); print matrix

    if names[j] == "LDA":
        lda_ap = metrics.precision_score(y_test, y_pred)
        lda_rec = metrics.recall_score(y_test, y_pred, average='weighted')
        lda_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    if names[j] == "SVM":
        svm_ap = metrics.precision_score(y_test, y_pred)
        svm_rec = metrics.recall_score(y_test, y_pred, average='weighted')
        svm_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    if names[j] == "DT-Gini":
        gini_ap = metrics.precision_score(y_test, y_pred)
        gini_rec = metrics.recall_score(y_test, y_pred, average='weighted')
        gini_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    if names[j] == "DT-ig":
        ig_ap = metrics.precision_score(y_test, y_pred)
        ig_rec = metrics.recall_score(y_test, y_pred, average='weighted')
        ig_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    if names[j] == "RandomForest":
        rf_ap = metrics.precision_score(y_test, y_pred)
        rf_rec = metrics.recall_score(y_test, y_pred, average='weighted')
        rf_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    j = j + 1

y_ax = np.arange(len(names))
precision = [lda_ap, svm_ap, gini_ap, ig_ap, rf_ap]
recall = [lda_rec, svm_rec, gini_rec, ig_rec, rf_rec]
f1_score = [lda_f1, svm_f1, gini_f1, ig_f1, rf_f1]

plt.bar(y_ax, precision, align="center")
plt.xticks(y_ax, names)
plt.ylabel("precision")
plt.show()

plt.bar(y_ax, recall, align="center")
plt.xticks(y_ax, names)
plt.ylabel("recall")
plt.show()

plt.bar(y_ax, f1_score, align="center")
plt.xticks(y_ax, names)
plt.ylabel("f1_score")
plt.show()