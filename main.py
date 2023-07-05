import warnings
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


mnist = fetch_openml('mnist_784',parser="auto")

x,y = mnist['data'].to_numpy() , mnist['target'].to_numpy()

digit = x[36001]

digit_image = digit.reshape(28,28)

# print(y[36000])

# plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.axis("off")

# plt.show()

x_train , x_test = x[:60000] , x[60000:]
y_train , y_test = y[60000]  , y[60000:]

shuffled = np.random.permutation(60000)
x_train , y_train = x[shuffled] , y[shuffled]

y_train= y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train_2 = (y_train==2)
y_test_2 = (y_test==2)
# print(y_train_2)


clf = LogisticRegression(tol=0.1)
clf.fit(x_train,y_train_2)

print(clf.predict([digit]))

accuracy = cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")
print(accuracy)