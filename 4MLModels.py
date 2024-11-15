import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("processed_data.csv")

offense_columns = [
    'offense_name_CyberCrime', 'offense_name_DrugCrime', 'offense_name_Fraud',
    'offense_name_Gambling', 'offense_name_OtherCrime', 'offense_name_PropertyCrime',
    'offense_name_SexCrime', 'offense_name_ViolentCrime', 'offense_name_WeaponCrime'
]

data['offense_name_combined'] = data[offense_columns].idxmax(axis=1)

min_samples = 1000
class_counts = data['offense_name_combined'].value_counts()
low_count_classes = class_counts[class_counts < min_samples].index.tolist()

data['offense_name_combined'] = data['offense_name_combined'].apply(
    lambda x: x if x not in low_count_classes else 'offense_name_Other'
)

X = data.drop(offense_columns + ['offense_name_combined'], axis=1)
y = data['offense_name_combined']

data_combined = pd.concat([X, y], axis=1)
data_combined = data_combined.dropna()
X = data_combined.drop('offense_name_combined', axis=1)
y = data_combined['offense_name_combined']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


def predict_sgd_svm(x_train, y_train, x_test=None):
    sgd_svm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
    sgd_svm.fit(x_train, y_train)
    if x_test is not None:
        y_prediction = sgd_svm.predict(x_test)
        return sgd_svm, y_prediction
    return sgd_svm

def predict_balanced_random_forest(x, Y):
    balanced_random_forest = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42,
        sampling_strategy='all',
        replacement=True,
        bootstrap=False,
        n_jobs=-1
    )
    balanced_random_forest.fit(x, Y)
    return balanced_random_forest


def predict_logistic_regression(x, Y, z=None):
    logistic_regression = LogisticRegression(solver='saga', max_iter=1000)
    logistic_regression.fit(x, Y)
    if z is not None:
        y_prediction = logistic_regression.predict(z)
        return logistic_regression, y_prediction
    return logistic_regression

def knn_algorithm(x, Y, z = 7):
    knn_prediction = KNeighborsClassifier(n_neighbors=z)
    knn_prediction.fit(x, Y)
    return knn_prediction

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    # plt.show()

print("Models before using SMOTE:")

# Logistic Regression
logistic_model, logistic_predictions = predict_logistic_regression(X_train, y_train, X_test)
print("\nLogistic Regression Model Performance:")
print(classification_report(y_test, logistic_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
pickle.dump(logistic_model, open("logistic_model_before_smote.pkl", "wb"))
plot_confusion_matrix(y_test, logistic_predictions, "Logistic Regression (Before SMOTE)")

# SGD-based SVM
sgd_svm_model, sgd_svm_predictions = predict_sgd_svm(X_train, y_train, X_test)
print("\nSGD-based SVM Model Performance:")
print(classification_report(y_test, sgd_svm_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, sgd_svm_predictions))
pickle.dump(sgd_svm_model, open("sgd_svm_model_before_smote.pkl", "wb"))
plot_confusion_matrix(y_test, sgd_svm_predictions, "SGD-based SVM (Before SMOTE)")

# KNN
knn_model = knn_algorithm(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
print("\nKNN Model Performance:")
print(classification_report(y_test, knn_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, knn_predictions))
pickle.dump(knn_model, open("knn_model_before_smote.pkl", "wb"))
plot_confusion_matrix(y_test, knn_predictions, "KNN (Before SMOTE)")

# Balanced Random Forest
rbf_model = predict_balanced_random_forest(X_train, y_train)
rbf_predictions = rbf_model.predict(X_test)
print("\nBalanced Random Forest Model Performance:")
print(classification_report(y_test, rbf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rbf_predictions))
pickle.dump(rbf_model, open("balanced_random_forest_before_smote.pkl", "wb"))
plot_confusion_matrix(y_test, rbf_predictions, "Balanced Random Forest (Before SMOTE)")

# Train and save models after using SMOTE
print("\n\n\nModels after using SMOTE:")

# Logistic Regression
logistic_model_smote, logistic_predictions_smote = predict_logistic_regression(X_train_resampled, y_train_resampled, X_test)
print("\nLogistic Regression Model Performance with SMOTE:")
print(classification_report(y_test, logistic_predictions_smote, zero_division=0))
print("Accuracy:", accuracy_score(y_test, logistic_predictions_smote))
pickle.dump(logistic_model_smote, open("logistic_model_after_smote.pkl", "wb"))
plot_confusion_matrix(y_test, logistic_predictions_smote, "Logistic Regression (After SMOTE)")


# SGD-based SVM
sgd_svm_model_smote = predict_sgd_svm(X_train_resampled, y_train_resampled)
sgd_svm_predictions_smote = sgd_svm_model_smote.predict(X_test)
print("\nSGD-based SVM Model Performance with SMOTE:")
print(classification_report(y_test, sgd_svm_predictions_smote, zero_division=0))
print("Accuracy:", accuracy_score(y_test, sgd_svm_predictions_smote))
pickle.dump(sgd_svm_model_smote, open("sgd_svm_model_after_smote.pkl", "wb"))
plot_confusion_matrix(y_test, sgd_svm_predictions_smote, "SGD-based SVM (After SMOTE)")


# KNN
knn_model_smote = knn_algorithm(X_train_resampled, y_train_resampled)
knn_predictions_smote = knn_model_smote.predict(X_test)
print("\nKNN Model Performance with SMOTE:")
print(classification_report(y_test, knn_predictions_smote, zero_division=0))
print("Accuracy:", accuracy_score(y_test, knn_predictions_smote))
pickle.dump(knn_model_smote, open("knn_model_after_smote.pkl", "wb"))
plot_confusion_matrix(y_test, knn_predictions_smote, "KNN (After SMOTE)")


# Balanced Random Forest
rbf_model_smote = predict_balanced_random_forest(X_train_resampled, y_train_resampled)
rbf_predictions_smote = rbf_model_smote.predict(X_test)
print("\nBalanced Random Forest Model Performance with SMOTE:")
print(classification_report(y_test, rbf_predictions_smote, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rbf_predictions_smote))
pickle.dump(rbf_model_smote, open("balanced_random_forest_after_smote.pkl", "wb"))
plot_confusion_matrix(y_test, rbf_predictions_smote, "Balanced Random Forest (After SMOTE)")
