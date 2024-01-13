from sklearn import tree
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np

# Creación de los datos
X, Y = make_blobs(n_samples=200, centers=4, random_state=6)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30)
plt.title("Datos originales")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
# Creación del árbol
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
plt.figure()
tree.plot_tree(clf)
plt.show()
#
DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        )
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30)
plt.show()
# Utilizar el árbol
print(clf.predict([[5.0, 1.0]]))
print(clf.predict([[-2.0, -1.0]]))
print(clf.predict([[6.0, -6.0]]))
