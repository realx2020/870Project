import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.layers import SpatialDropout1D
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics

"""=================================================================================================================="""

'''Section 1: Data Pre-processing'''
data_calls = pd.read_csv("calls.csv")
data_labels = pd.read_csv("labels.csv")
data_all = data_calls.join(data_labels)
#print(data_calls.head())
#print(data_labels.head())
#print(data_all.head())

"""
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(6)
ax = sns.countplot(x=data_labels["API_Labels"], data=data_labels, width=0.7)
ax.bar_label(ax.containers[0])
plt.xlabel('Software Classification')
plt.ylabel('Samples')
plt.title('Distribution')
plt.savefig("distribution.png")
plt.show()
"""


data_labels.loc[data_labels["API_Labels"] == "Spyware", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Downloader", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Trojan", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Worms", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Adware", "API_Labels"] = "Other" #note
data_labels.loc[data_labels["API_Labels"] == "Dropper", "API_Labels"] = "Other" #note
data_labels.loc[data_labels["API_Labels"] == "Virus", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Backdoor", "API_Labels"] = "Other"
data_labels.loc[data_labels["API_Labels"] == "Goodware", "API_Labels"] = "Focus" #note


'''Section 2: Splitting Train/Test Data'''
X = data_calls
Y = data_labels["API_Labels"].astype('category').cat.codes
#Y = data_all["API_Labels"].astype('category').cat.codes
#X = data_all.drop(columns=["API_Labels"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
#print('X train shape: ', X_train.shape)
#print('Y train shape: ', Y_train.shape)
#print('X test shape: ', X_test.shape)
#print('Y test shape: ', Y_test.shape)


'''Section 3: Machine Learning Algorithms'''

# Random Forest Classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
random_forest_score = random_forest.score(X_test, Y_test)
print("The score of Random Forest is", random_forest_score * 100)

"""
# Naive Bayes
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(X_train, Y_train)
gaussian_naive_bayes_score = gaussian_naive_bayes.score(X_test, Y_test)
print("The score of Gaussian Naive Bayes is", gaussian_naive_bayes_score * 100)
"""

"""
# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
logistic_regression_score = logistic_regression.score(X_test, Y_test)
print("The score of Logistic Regression is", logistic_regression_score * 100)
"""

# Decision Tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
decision_tree_score = decision_tree.score(X_test, Y_test)
print("The score of Decision Tree is", decision_tree_score * 100)

# Support Vector Machine
support_vector_machine = svm.SVC(probability=True)
support_vector_machine.fit(X_train, Y_train)
support_vector_machine_score = support_vector_machine.score(X_test, Y_test)
print("The score of Support Vector Machine is", support_vector_machine_score * 100)

"""
# Multi-layer Perceptron
multi_layer_perceptron = MLPClassifier()
multi_layer_perceptron.fit(X_train, Y_train)
multi_layer_perceptron_score = multi_layer_perceptron.score(X_test, Y_test)
print("The score of Multi-layer Perceptron is", multi_layer_perceptron_score * 100)
"""

"""
# Artificial Neural Network
artificial_neural_network = tf.keras.models.Sequential()
artificial_neural_network.add(tf.keras.layers.Dense(units=6, activation="relu"))
artificial_neural_network.add(tf.keras.layers.Dense(units=6, activation="relu"))
artificial_neural_network.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
artificial_neural_network.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
history1 = artificial_neural_network.fit(X_train, Y_train,validation_split=0.25, batch_size=32, epochs = 100)
"""

"""
# Long Short-term Memory Model
max_words = 800
max_len = 100

def lstm_model(act_func="softsign"):
    model = Sequential()
    model.add(Embedding(max_words, 300, input_length=max_len))
    model.add(SpatialDropout1D(0.1))
    model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True, activation=act_func))
    model.add(LSTM(32, dropout=0.1, activation=act_func, return_sequences=True))
    model.add(LSTM(32, dropout=0.1, activation=act_func))
    model.add(Dense(128, activation=act_func))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation=act_func))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation=act_func))
    model.add(Dropout(0.1))
    model.add(Dense(1, name='out_layer', activation="linear"))
    return model

long_short_term_memory_model = lstm_model()
#print(long_short_term_memory_model.summary())
long_short_term_memory_model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])
long_short_term_memory_model.fit(X_train, Y_train, batch_size=1000, epochs=10, validation_data=(X_test, Y_test), verbose=1)
"""

"""
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig("accuracy.png")
plt.show()
"""


cm = confusion_matrix(Y_test, support_vector_machine.predict(X_test))
plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
plt.savefig("confusion_matrix.png")
plt.show()



y_pred_proba = random_forest.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr, tpr, label="AUC = "+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

