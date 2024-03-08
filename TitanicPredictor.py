# Import necessary libraries
from sklearn import svm, preprocessing, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

#put your data path here i.e. User/Downloads/titanic
datapath ="titanic"

train_data = pd.read_csv(f'{datapath}/train.csv')
test_data = pd.read_csv(f'{datapath}/test.csv')
# Preprocess data
def preprocess_data(data):
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 0, 'IsAlone'] = 1
    return data

###NEURAL NETWORK IMPLEMENTATION###

# Load and preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Split into features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Create a pipeline
pipe = make_pipeline(preprocessing.StandardScaler(), svm.SVC())

#this gridsearch will output the optimal hyperparameter tunings: C = 1, gamma = 0.1 and kernel ='rbf'
# Define parameter grid 
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [1, 0.1, 0.01, 0.001], 'svc__kernel': ['rbf']}

# Create a GridSearchCV object
grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=5), refit=True, verbose=0)
grid.fit(X, y)

# Print best parameters
print(grid.best_params_)

# Cross validation
scores_svm = model_selection.cross_val_score(grid, X, y, cv=StratifiedKFold(n_splits=5))
print(f'Cross-validation accuracy: {scores_svm.mean()*100}%')

# Predict on test data
predictions_svm = grid.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions_svm
})
submission.to_csv('SVMsubmission.csv', index=False)

###NEURAL NETWORK IMPLEMENTATION###

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



# Compile the model
# model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluate the model
loss, accuracy_nn = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {accuracy_nn*100}%')

# Predict on test data
predictions_nn = model.predict(test_data)
predictions_nn = [1 if p > 0.5 else 0 for p in predictions_nn]

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions_nn
})
submission.to_csv('NNsubmission.csv', index=False)

###RANDOM FOREST IMPLEMENTATION###

# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100,max_features='log2',max_depth=7)

# # Define parameter grid
# param_grid = {'n_estimators': [100, 200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [6,7,8]}

# # Create a GridSearchCV object
# grid = GridSearchCV(rfc, param_grid, cv=StratifiedKFold(n_splits=5), refit=True, verbose=0)

# grid.fit(X, y)

# # Print best parameters
# print(grid.best_params_)

# Cross validation
scores_tree = cross_val_score(rfc, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1, scoring='accuracy')
print(f'Cross-validation accuracy: {scores_tree.mean()*100}%')
rfc.fit(X,y)

# Predict on test data
predictions_trees = rfc.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions_trees
})
submission.to_csv('RFCsubmission.csv', index=False)

svm_accuracy = scores_svm.mean()
rf_accuracy = scores_tree.mean()
nn_accuracy = accuracy_nn

# Model names
models = ['SVM', 'Random Forest', 'Neural Network']

# Accuracy scores
accuracy_scores = [svm_accuracy, rf_accuracy, nn_accuracy]
test_accuracy_scores = [0.78468, 0.77033, 0.72009]

# Create bar plot for training accuracy
plt.bar(models, accuracy_scores, color=['blue'], width=0.4, align='center', label='Training Accuracy')

# Create bar plot for test accuracy
plt.bar(models, test_accuracy_scores, color=['lightblue'], width=0.4, align='edge', label='Test Accuracy')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Performance')
plt.ylim([0.5, 1])
plt.legend()  # Add a legend
plt.show()