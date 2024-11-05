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

importances = brf_model.feature_importances_
feature_names = X_train_resampled.columns

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
#---------------------- TRAIN THE INITIAL MODEL ----------------------#

#----------------------- FEATURE REDUCTION LOOP ----------------------#
num_features_list = []
accuracy_list = []
macro_f1_list = []

max_features = len(feature_importance_df)

for n_features in tqdm(range(max_features, 0, -1)):
    top_features = feature_importance_df['feature'].iloc[:n_features].tolist()
    X_train_resampled_reduced = X_train_resampled[top_features]
    X_test_reduced = X_test[top_features]
    
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
    
    acc = accuracy_score(y_test, y_pred_brf_reduced)
    report = classification_report(y_test, y_pred_brf_reduced, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    num_features_list.append(n_features)
    accuracy_list.append(acc)
    macro_f1_list.append(macro_f1)

f = open("RandomForrestResults.txt", "w")
for i in range(len(num_features_list)):
    f.write(f"Number of Features: {num_features_list[i]}\n")
    f.write(f"Accuracy: {accuracy_list[i]}\n")
    f.write(f"Macro F1-score: {macro_f1_list[i]}\n\n")
#----------------------- FEATURE REDUCTION LOOP ----------------------#

#-------------------------- PLOT RESULTS -----------------------------#
plt.figure(figsize=(12, 6))
plt.plot(num_features_list, accuracy_list, label='Accuracy')
plt.plot(num_features_list, macro_f1_list, label='Macro F1-score')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('Model Performance vs. Number of Features')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()
#-------------------------- PLOT RESULTS -----------------------------#

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)






