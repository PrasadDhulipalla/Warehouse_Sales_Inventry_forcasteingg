import joblib
import pandas as pd
import matplotlib.pyplot as plt

def forecast_sales(df, models):
    for item, model in models.items():
        item_df = df[df['Item'] == item].copy()
        item_df['Day'] = (item_df['Date'] - item_df['Date'].min()).dt.days
        future_days = pd.DataFrame({'Day': range(item_df['Day'].max() + 1, item_df['Day'].max() + 31)})
        future_pred = model.predict(future_days)
        plt.figure(figsize=(10, 4))
        plt.plot(item_df['Date'], item_df['Sales'], label='Historical Sales')
        future_dates = pd.date_range(start=item_df['Date'].max() + pd.Timedelta(days=1), periods=30)
        plt.plot(future_dates, future_pred, label='Forecasted Sales', linestyle='--')
        plt.title(f"Sales Forecast for {item}")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"models/{item}_forecast.png")
        plt.close()
