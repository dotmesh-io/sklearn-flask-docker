import os
import sklearn
from sklearn import svm
from sklearn import datasets
from pickle import dump
import json

clf = svm.SVC(gamma='scale', probability=True)
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

dump(clf, open("model.joblib", "wb"))
with open("classes.json", "w") as f:
    f.write(json.dumps({"0": "Setosa",
                        "1": "Versicolour",
                        "2": "Virginica"}))
