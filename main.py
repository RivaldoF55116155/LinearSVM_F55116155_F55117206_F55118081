from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


cancer_data = datasets.load_breast_cancer()

X_train, X_test, y_train, y_teat = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4, random_state=209)
# generate the module
cls = svm.SVC(kernel="linear")
# train the model
cls.fit(X_train, y_train)
# predict the response
pred = cls.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_test, y_pred=pred))
# precision aoere
print("precision:", metrics.precision_score(y_test, y_pred=pred))
# recall score
print ("recall", metrics.recall_score(y_test, y_pred=pred))
print(metrics.classification_report(y_test, y_pred=pred))