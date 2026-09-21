import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv')

X = df.drop('actual_runtime').values
y = df['actual_runtime'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)