import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

np.random.seed(42)
n_samples = 395

data = {
    'studytime': np.random.randint(1, 5, n_samples),
    'failures': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
    'absences': np.random.randint(0, 20, n_samples),
    'G1': np.random.randint(5, 20, n_samples),
    'G2': np.random.randint(5, 20, n_samples),
}

df = pd.DataFrame(data)
df['G3'] = df['G1'] * 0.5 + df['G2'] * 0.4 - df['failures'] * 1.5 + df['studytime'] * 0.5 + np.random.normal(0, 1.5, n_samples)
df['G3'] = np.clip(df['G3'], 0, 20)

X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)
lr_pred = lr_model.predict(X_test_scaled)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, Y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

print("--- Linear Regression ---")
print("MSE:", metrics.mean_squared_error(Y_test, lr_pred))
print("MAE:", metrics.mean_absolute_error(Y_test, lr_pred))
print("R2 Score:", lr_model.score(X_test_scaled, Y_test))

print("\n--- Ridge Regression ---")
print("MSE:", metrics.mean_squared_error(Y_test, ridge_pred))
print("MAE:", metrics.mean_absolute_error(Y_test, ridge_pred))
print("R2 Score:", ridge_model.score(X_test_scaled, Y_test))

cv_scores = cross_val_score(lr_model, X_train_scaled, Y_train, cv=5, scoring='r2')
print("\nCross-Validation R2 Scores:", cv_scores)
print("Mean CV R2 Score:", np.mean(cv_scores))

sample_student = np.array([[2, 0, 4, 12, 14]])
sample_student_scaled = scaler.transform(sample_student)
predicted_grade = lr_model.predict(sample_student_scaled)
print("\nPredicted Final Grade (G3) for Sample Student:", predicted_grade[0])