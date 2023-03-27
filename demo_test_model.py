# Sample Execution of predictCompressiveStrength function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Import the Group12_best_model Prediction Function
from group12_best_model import predictCompressiveStrength

# Load and Extract the X,Y From the Blinded Test Data file
data_dir = "./data/"
p_btest_data = data_dir+"test_blind.csv"
blind_test_data = pd.read_csv(p_btest_data)

X_bl_test = blind_test_data.iloc[:, :-1]
Y_bl_test = blind_test_data.iloc[:, -1]

# Predict the Compressive Strength for the Blind Test X Data
y_pred_best = predictCompressiveStrength(X_bl_test, data_dir)

# Evaluate the final Best regression model on the Blind test Data using R2 metrics
rse_best = np.sqrt(mean_squared_error(Y_bl_test, y_pred_best))
r2_best = r2_score(Y_bl_test, y_pred_best)

print("Best - Residual Standard Rrror (RSE):", rse_best)
print("Best - R2 score:", r2_best)
