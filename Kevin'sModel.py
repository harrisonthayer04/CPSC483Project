from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("processed_data.csv")

offense_columns = [
    'offense_name_CyberCrime', 'offense_name_DrugCrime', 'offense_name_Fraud',
    'offense_name_Gambling', 'offense_name_OtherCrime', 'offense_name_PropertyCrime',
    'offense_name_SexCrime', 'offense_name_ViolentCrime', 'offense_name_WeaponCrime'
]

# ----------------------- CREATE THE DATAFRAME ------------------------#

min_samples = 1000
class_counts = data.value_counts()
low_count_classes = class_counts[class_counts < min_samples].index.tolist()

X = data.drop(offense_columns, axis=1)
X.fillna(0, inplace=True)

y = data.drop(offense_columns, axis=1)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------ INPUTING -----------------------------#

def predict_linear_regression(x, Y, z):
    linear_regression = LinearRegression()
    linear_regression.fit(x, Y)
    y_prediction = linear_regression.predict(z)
    return y_prediction

linear_model = predict_linear_regression(X_train, y_train, X_test)
linear_modely = predict_linear_regression(X_train, y_train, y_test)

# --------------------------- VISUALIZATIONS --------------------------#
plt.scatter(X, y, color="blue")
plt.plot(linear_model, linear_modely, color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression")
plt.show()
