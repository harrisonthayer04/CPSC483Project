import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

# ----------------------- CREATE THE DATAFRAME ------------------------#
df = pd.read_csv("processed_data.csv")
df.head()

features = ['population_group_class_Medium_City', 'population_group_class_Small_City']
x2 = df[features]
y2 = df.population_group_class_Medium_City
X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.25, random_state=16)

# ------------------------------ INPUTING -----------------------------#

def predict_logistic_regression(X, y, z):
    logistic_regression = LogisticRegression(random_state=16)
    logistic_regression.fit(X, y)
    y_prediction = logistic_regression.predict(z)
    return y_prediction

log = predict_logistic_regression(X_train, y_train, X_test)

# --------------------------- VISUALIZATIONS --------------------------#
matrix1 = metrics.confusion_matrix(y_test, log)
names=[0,1]
fig, axis = plt.subplots()
ticks = np.arange(len(names))
plt.xticks(ticks, names)
plt.yticks(ticks, names)
# create heatmap
sns.heatmap(pd.DataFrame(matrix1), annot=True, cmap="YlGnBu" ,fmt='g')
axis.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
