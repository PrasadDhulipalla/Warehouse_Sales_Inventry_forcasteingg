from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os

def train_model(df):
    models = {}
    for item in df['Item'].unique():
        item_df = df[df['Item'] == item].copy()
        item_df['Day'] = (item_df['Date'] - item_df['Date'].min()).dt.days
        X = item_df[['Day']]
        y = item_df['Sales']
        model = LinearRegression()
        model.fit(X, y)
        models[item] = model
        joblib.dump(model, f"models/{item}_model.pkl")
    return models
