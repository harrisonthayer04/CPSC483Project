import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

#--------------------- LOAD AND ENCODE THE DATA ----------------------#
data = pd.read_csv("processed_data.csv")

offense_columns = [
    'offense_name_CyberCrime', 'offense_name_DrugCrime', 'offense_name_Fraud',
    'offense_name_Gambling', 'offense_name_OtherCrime', 'offense_name_PropertyCrime',
    'offense_name_SexCrime', 'offense_name_ViolentCrime', 'offense_name_WeaponCrime'
]

data['offense_name_combined'] = data[offense_columns].idxmax(axis=1)
#--------------------- LOAD AND ENCODE THE DATA ----------------------#

#--------------------- COMBINE UNCOMMON CLASSES ----------------------#
min_samples = 1000
class_counts = data['offense_name_combined'].value_counts()
low_count_classes = class_counts[class_counts < min_samples].index.tolist()

data['offense_name_combined'] = data['offense_name_combined'].apply(
    lambda x: x if x not in low_count_classes else 'offense_name_Other'
)

X = data.drop(offense_columns + ['offense_name_combined'], axis=1)
y = data['offense_name_combined']
#--------------------- COMBINE UNCOMMON CLASSES ----------------------#

#-------------------------- SPLIT THE DATA ---------------------------#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#-------------------------- SPLIT THE DATA ---------------------------#

#---------------------------- APPLY SMOTE ----------------------------#
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#---------------------------- APPLY SMOTE ----------------------------#

#---------------------- TRAIN THE INITIAL MODEL ----------------------#
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

pickle.dump(brf_model, open("InitialBRFModel", 'wb'))

importances = brf_model.feature_importances_
feature_names = X_train_resampled.columns

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
#---------------------- TRAIN THE INITIAL MODEL ----------------------#

#---------------------- FEATURE REDUCTION MODEL ----------------------#
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
pickle.dump(brf_model_reduced, open("NewBRFModel", 'wb'))
#---------------------- FEATURE REDUCTION MODEL ----------------------#


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)






