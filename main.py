from src.data_loader import load_data
from src.model import train_model
from src.predict import forecast_sales

def main():
    df = load_data("data/warehouse_sales_large.csv")
    models = train_model(df)
    forecast_sales(df, models)

if __name__ == "__main__":
    main()
