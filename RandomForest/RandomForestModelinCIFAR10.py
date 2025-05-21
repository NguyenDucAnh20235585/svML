#Tran Ngoc Minh - 20235608
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.datasets import cifar10
from keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(x_train, y_train.ravel())  


y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average= 'weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

print(classification_report(y_test, y_pred))

y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
test_f1_score = f1_score(y_test, y_test_pred, average='weighted')

plt.figure(figsize=(2, 1))
plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy])
plt.title('Accuracy of Random Forest on CIFAR-10')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(2, 1))
plt.bar(['Training Precision', 'Test Precision'], [train_precision, test_precision])
plt.title('Precision of Random Forest on CIFAR-10')
plt.ylabel('Precision')
plt.show()

plt.figure(figsize=(2, 1))
plt.bar(['Training Recall', 'Test Recall'], [train_recall, test_recall])
plt.title('Recall of Random Forest on CIFAR-10')
plt.ylabel('Recall')
plt.show()

plt.figure(figsize=(2, 1))
plt.bar(['Training F1 Score', 'Test F1 Score'], [train_f1_score, test_f1_score])
plt.title('F1 Score of Random Forest on CIFAR-10')
plt.ylabel('F1 Score')
plt.show()
