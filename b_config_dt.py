import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier

###############################     DATASETS     ##########################
datasets = np.genfromtxt("spambase.data", delimiter=",")
observation, features = datasets.shape

X = []; y = []
dt_cri = ["gini", "entropy"]
dt_leaf = [2, 5, 10, 20]
f_gini = []; f_ig = []

file1 = open("spambase.data", 'r')
for data in file1:
    rc = data.split(',')
    X.append(rc[:-1])
    y.append(rc[features-1].strip('\n'))

linearly_separable = (X, y)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

for leaf in dt_leaf:

    classifier_dt_gini = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=leaf)
    classifier_dt_ig = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=leaf)
    f_msr_dt_gini = cross_validate(classifier_dt_gini, X, y, scoring=['f1_macro'], cv=10, return_train_score=False)
    f_msr_dt_ig = cross_validate(classifier_dt_ig, X, y, scoring=['f1_macro'], cv=10, return_train_score=False)

    tmp = 0
    for Fscore in f_msr_dt_gini['test_f1_macro']:
        tmp = tmp + Fscore
    f_avg_dt_gini = tmp/10
    f_gini.append(f_avg_dt_gini)

    tmp = 0
    for Fscore in f_msr_dt_ig['test_f1_macro']:
        tmp = tmp + Fscore
    f_avg_dt_ig = tmp/10
    f_ig.append(f_avg_dt_ig)

    # print "\n test_f1_macro :", f_msr_dt_gini['test_f1_macro']
    print "Criterion= gini, leaf=", leaf, "\t F-avg", f_avg_dt_gini
    print "Criterion= ig, leaf=", leaf, "\t F-avg", f_avg_dt_ig

plt.plot(dt_leaf, f_gini, marker="o")
plt.plot(dt_leaf, f_ig, marker="x")
plt.ylabel("F Measure")
plt.xlabel("K-values")

plt.show()
