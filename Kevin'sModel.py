def predict_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
    logistic_regression = LogisticRegression(random_state=10)
    logistic_regression.fit(X_train, y_train)
    y_prediction = logistic_regression.predict(X_test)
    return y_prediction

log = predict_logistic_regression(df.index, cols)