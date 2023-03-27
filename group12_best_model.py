# Import necessary libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def predictCompressiveStrength(Xtest, data_dir):

  # Load the Data
  #############################################
  p_train_data = data_dir+"train.csv"
  p_test_data = data_dir+"test.csv"

  train_data = pd.read_csv(p_train_data)
  test_data = pd.read_csv(p_test_data)

  # Extract the Data
  #############################################
  # Extract the Design Matrix of 8 features from the Data Set
  X_train = train_data.iloc[:, :-1] # Training
  X_test = test_data.iloc[:, :-1]   # Test

  # Extract Output from the Data Set
  Y_train = train_data.iloc[:, -1]  # Training
  Y_test = test_data.iloc[:, -1]    # Test

  # Create a GradientBoostingRegressor model and Tune its Hyper-Parameter
  # ############################################################################ 
  model = GradientBoostingRegressor()

  # Define the hyperparameters to be tuned
  param_grid = {'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 7]}
                
  # param_grid = {'n_estimators': [100, 500, 1000],
  #               'learning_rate': [0.01, 0.1, 0.5],
  #               'max_depth': [3, 5, 10]}
                
  # Use GridSearchCV to find the best hyperparameters
  grid_search = GridSearchCV(model, param_grid, cv=5)
  grid_search.fit(X_train, Y_train)

  # Print the best hyperparameters
  print("Best Hyperparameters:", grid_search.best_params_)

  # Create the Final GradientBoostingRegressor model using "Best" Hyper-Parameters
  # ############################################################################ 
  final_model = GradientBoostingRegressor(**grid_search.best_params_)

  # Combine Training data and the Validation/Test Data
  X_Comb = pd.concat([X_train, X_test], ignore_index=True)
  Y_Comb = pd.concat([Y_train, Y_test], ignore_index=True)

  # Train the final model on the Combined training and Validation Data
  final_model.fit(X_Comb, Y_Comb)

  # Make predictions on the Blinded test data
  y_pred = final_model.predict(Xtest)

  # Return the predictions on the Blinded Test Data
  return y_pred