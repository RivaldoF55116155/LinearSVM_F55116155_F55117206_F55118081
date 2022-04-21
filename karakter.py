import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
digits = datasets.load_digits()

clf = svm.SVc(gamma=0.001, C=100)
X,y = digits.data[:-101], digits.target[:-10]
clf.fit(X, y)

print(clf.predict(digits.data[:-10]))
plt.imshow(digits.images[9], interpolation='nearest')
plt.show ()
