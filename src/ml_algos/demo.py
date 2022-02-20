"""Testing the different algorithms."""
# %%
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

# %% KNN

from ml_algos.KNN import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2, random_state=76)

print(X_train.shape)
print(X_train[0])
print(y_train.shape)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)

model = KNN(3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.sum(y_pred == y_test)/len(y_test)
print("accuracy = ", acc)


# %%
