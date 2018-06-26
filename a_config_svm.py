import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm

###############################     DATASETS     ##########################
datasets = np.genfromtxt("spambase.data", delimiter=",")
observation, features = datasets.shape
X = []; y = []; f = []
c_val = [0.01, 0.1, 1, 10, 100]

file1 = open("spambase.data", 'r')
for data in file1:
    rc = data.split(',')
    X.append(rc[:-1])
    y.append(rc[features-1].strip('\n'))

linearly_separable = (X, y)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

for c in c_val:
    tmp = 0.0
    classifier_svm = svm.LinearSVC(C=c).fit(X_train, y_train)
    F_msr_svm = cross_validate(classifier_svm, X, y, scoring=['f1_macro'], cv=10, return_train_score=False)

    for Fscore in F_msr_svm['test_f1_macro']:
        tmp = tmp + float(Fscore)
    f_avg_svm = float(tmp)/10
    f.append(float(f_avg_svm))

    # print "\t F_msr['test_f1_macro'] :", F_msr_svm['test_f1_macro']
    print "C :", c, "\t F-avg", f_avg_svm

plt.plot(c_val, f, marker='x')
plt.ylabel("F Measure")
plt.xlabel("C-values")
plt.show()
