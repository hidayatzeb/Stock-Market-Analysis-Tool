import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class StockMarketAnalysisTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Analysis Tool")

        self.stock_symbol_label = tk.Label(self.root, text="Stock Symbol:")
        self.stock_symbol_label.pack()

        self.stock_symbol_entry = tk.Entry(self.root)
        self.stock_symbol_entry.pack()

        self.button_fetch_data = tk.Button(self.root, text="Fetch Data", command=self.fetch_data)
        self.button_fetch_data.pack()

        self.result_label = tk.Label(self.root, text="Result:")
        self.result_label.pack()

        self.result_text = tk.Text(self.root, height=30, width=120)
        self.result_text.pack()

    def fetch_data(self):
        stock_symbol = self.stock_symbol_entry.get().upper()
        if stock_symbol == "":
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return

        try:
            data = yf.download(stock_symbol)
            # Perform data preprocessing
            data = self.preprocess_data(data)

            # Train the model
            model = self.train_model(data)

            # Display historical data and predictions
            self.display_results(data, model)
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def preprocess_data(self, data):
        # Handle missing or incorrect data
        data = data.dropna()

        return data

    def train_model(self, data):
        # Prepare the data
        X = data.index.astype(np.int64) // 10 ** 9  # Convert date to UNIX timestamp
        y = data['Close']

        # Train the model
        model = LinearRegression()
        model.fit(X.values.reshape(-1, 1), y)

        return model

    def display_results(self, data, model):
        self.result_text.delete(1.0, tk.END)

        # Display historical data
        self.result_text.insert(tk.END, "Historical Data:\n")
        self.result_text.insert(tk.END, str(data) + "\n\n")

        # Display predictions
        self.result_text.insert(tk.END, "Predictions:\n")
        prediction_date = data.index[-1].strftime("%Y-%m-%d")
        prediction = model.predict([[data.index[-1].timestamp()]])
        self.result_text.insert(tk.END, f"Prediction for {prediction_date}: {prediction[0]:.2f}\n")

        # Plot historical data and predictions
        self.plot_data(data, model)

    def plot_data(self, data, model):
        plt.figure(figsize=(8, 6))
        plt.plot(data.index, data['Close'], label='Historical Data')

        # Generate date range for predictions
        start_date = data.index[0]
        end_date = data.index[-1]
        date_range = pd.date_range(start=start_date, end=end_date)

        # Generate predictions for the date range
        predictions = model.predict([[d.timestamp()] for d in date_range])

        plt.plot(date_range, predictions, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Historical Data and Predictions')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = StockMarketAnalysisTool(root)
    root.mainloop()
