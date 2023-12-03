import pandas as pd
from sklearn import linear_model, naive_bayes
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics


datasetFor2000 = pd.read_csv("dataset-of-00s.csv")
datasetFor2010= pd.read_csv("dataset-of-10s.csv")
x=pd.concat([datasetFor2000,datasetFor2010])


y=x["target"]
x=x.drop("target",axis=1)
x=x.drop("track",axis=1)
x=x.drop("artist",axis=1)
c = KFold(n_splits=5, random_state=1, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)


print("Decision Tree")
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print("Accuracy with training 80% of the data and testing with 20% ",clf.score(x_test,y_test))
scores = cross_val_score(clf, x, y, scoring='accuracy', cv=c, n_jobs=-1)
dt_y_pred = clf.predict(x_test)
dt_fpr, dt_tpr, threshold = metrics.roc_curve(y_test, dt_y_pred)
dt_roc_auc = metrics.auc(dt_fpr, dt_tpr)
print(scores,dt_roc_auc)
print(metrics.confusion_matrix(y_test,dt_y_pred))

print("Logistic Regression")
lr_clf = linear_model.LogisticRegression()
lr_clf.fit(x_train, y_train)
print("Accuracy with training 80% of the data and testing with 20% ",lr_clf.score(x_test,y_test))
scores = cross_val_score(lr_clf, x, y, scoring='accuracy', cv=c, n_jobs=-1)
lr_y_pred = clf.predict(x_test)
lr_y_pred =lr_clf.predict(x_test)
lr_fpr, lr_tpr, threshold1 = metrics.roc_curve(y_test, lr_y_pred)
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)
print(scores,lr_roc_auc)


print("KNN")
k_values = []
accuracy_scores = []
k_range = range(1, 300,10)

for k in k_range:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = knn_classifier.score(x_test,y_test)
    k_values.append(k)
    accuracy_scores.append(accuracy)

plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. k Value for k-NN')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid()
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=170)
knn_classifier.fit(x_train, y_train)
scores = cross_val_score(knn_classifier, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print(scores)
knn_y_pred = knn_classifier.predict(x_test)
# knn_fpr, knn_tpr, threshold = metrics.roc_curve(y_test, knn_y_pred)
# knn_roc_auc = metrics.auc(knn_fpr, knn_tpr)
# print(knn_roc_auc)


print("Naive Bayes")
NB_model = naive_bayes.GaussianNB()
NB_model.fit(x_train, y_train)
print(NB_model.score(x_test,y_test))
nb_y_pred=NB_model.predict(x_test)
# nb_fpr, nb_tpr, threshold = metrics.roc_curve(y_test, nb_y_pred)
# nb_roc_auc = metrics.auc(nb_fpr, nb_tpr)

print("Linear Discriminant")
LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(x_train, y_train)
print(LDA_model.score(x_test, y_test))
lda_y_pred=LDA_model.predict(x_test)
scores = cross_val_score(LDA_model, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print(scores)
lda_fpr, lda_tpr, threshold = metrics.roc_curve(y_test, lda_y_pred)
lda_roc_auc = metrics.auc(lda_fpr, lda_tpr)
print(metrics.confusion_matrix(y_test,lda_y_pred))

plt. figure(figsize=(8, 6))
# plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_roc_auc:.2f})')
# plt.plot(knn_fpr, knn_tpr, label=f'KNNTree (AUC = {knn_roc_auc:.2f})')
# plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_roc_auc:.2f})')
plt.plot(lda_fpr, lda_tpr, label=f'Linear Discriminant (AUC = {lda_roc_auc:.2f})')
plt. plot ([0, 1], [0, 1], 'k--')
plt.ylabel( 'True Positive Rate')
plt.xlabel("False Positive Rate")
plt. title( 'ROC Curves')
plt. legend (loc='best')
plt.grid (True)
plt.show()
