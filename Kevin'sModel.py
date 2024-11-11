import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle

data = pd.read_csv("processed_data.csv")

offense_columns = [
    'offense_name_CyberCrime', 'offense_name_DrugCrime', 'offense_name_Fraud',
    'offense_name_Gambling', 'offense_name_OtherCrime', 'offense_name_PropertyCrime',
    'offense_name_SexCrime', 'offense_name_ViolentCrime', 'offense_name_WeaponCrime'
]

X_linear = data.drop(offense_columns, axis=1)
X_linear.fillna(0, inplace=True)

y_linear = data.drop(offense_columns, axis=1)
y_linear.fillna(0, inplace=True)

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

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

brf_model = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    sampling_strategy='all',
    replacement=True,
    bootstrap=False,
    n_jobs=-1
)
brf_model.fit(X_train_resampled, y_train_resampled)
y_pred_brf = brf_model.predict(X_test)
print("\nBalanced Random Forest Model Performance:")
print(classification_report(y_test, y_pred_brf, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred_brf))
pickle.dump(brf_model, open("InitialBRFModel.pickle", 'wb'))

importances = brf_model.feature_importances_
feature_names = X_train_resampled.columns

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

def predict_linear_regression(x, Y):
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x, Y)
    return linear_regression

def predict_logistic_regression(x, Y, z):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x, Y)
    y_prediction = logistic_regression.predict(z)
    return y_prediction

def knn_algorithm(x, Y, z):
    knn_prediction = KNeighborsClassifier(n_neighbors=z)
    knn_prediction.fit(x, Y)
    return knn_prediction

linear = predict_linear_regression(X_train_lin, y_train_lin)

print("\nLinear Regression Model Performance:")
print('Coefficients: ', linear.coef_)
print('Variance score: {}'.format(linear.score(X_test_lin, y_test_lin)))

logistic_model = predict_logistic_regression(X_train, y_train, X_test)

print("\nLogistic Regression Model Performance:")
print(classification_report(y_test, logistic_model, zero_division=0))
print("Accuracy:", accuracy_score(y_test, logistic_model))

knn_model = knn_algorithm(X_train, y_train, 7)

print("\nKNN Model Performance:")
print(classification_report(y_test, knn_model.predict(X_test), zero_division=0))
print("Accuracy:", accuracy_score(y_test, knn_model.predict(X_test)))

least_important_features = feature_importance_df.tail(4)['feature'].tolist()
X_train_resampled_reduced = X_train_resampled.drop(columns = least_important_features)
X_test_reduced = X_test.drop(columns = least_important_features)
brf_model_reduced = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    sampling_strategy='all',
    replacement=True,
    bootstrap=False,
    n_jobs=-1
)
brf_model_reduced.fit(X_train_resampled_reduced, y_train_resampled)
y_pred_brf_reduced = brf_model_reduced.predict(X_test_reduced)

print("\nBalanced Random Forest Model Performance after Removing 4 Least Important Features:")
print(classification_report(y_test, y_pred_brf_reduced, zero_division=0))
accuracy_after = accuracy_score(y_test, y_pred_brf_reduced)
print("Accuracy:", accuracy_after)
pickle.dump(brf_model_reduced, open("NewBRFModel.pickle", 'wb'))
