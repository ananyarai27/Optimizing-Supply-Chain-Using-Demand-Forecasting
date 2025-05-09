# 📦 Supply Chain Demand Forecasting Dashboard
A Streamlit-powered interactive dashboard to forecast product demand and optimize supply chain operations using machine learning and time-series analysis.

# 🚀 Features
📁 Upload your own sales CSV file (week, units_sold, store_id, sku_id)

🔍 Filter by SKU, store, and date range

📈 Forecast demand using:

Rolling Mean (baseline)

ARIMA (time-series)

Random Forest (machine learning)

📉 View forecast accuracy metrics (MAE, RMSE, MAPE)

📦 Simulate inventory needs with safety stock and lead time

🔥 Highlight promotion weeks (is_featured_sku, is_display_sku)

# 📊 Download results as CSV or PDF

# 📊 Sample Use Case
"This tool helps supply chain managers accurately predict future demand, identify trends, and simulate inventory decisions — ultimately reducing overstocking, understocking, and cost."

# 🧰 Tech Stack
Python 🐍

Streamlit 🚪

Pandas / NumPy 📊

Scikit-learn / ARIMA 📈

Matplotlib / PdfPages 📄

# 📝 Installation
Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run Demand_Forecasting.py
📂 Sample Data Format:

Your CSV should include at least these columns:

week	store_id	sku_id	units_sold	is_featured_sku	is_display_sku
17/01/11	8091	216418	20	0	0

📌 week should be in YY/MM/DD format (e.g., 17/01/11)

# 📥 Output
Forecast table (actual vs forecast)

Charts (forecast line chart, bar chart with peak highlight)

Feature importance (for Random Forest)

CSV + PDF export

# 💡 Future Improvements
Add Prophet / LSTM models

Real-time deployment via Streamlit Cloud or Heroku

Inventory optimization logic (EOQ, reorder points)

📜 License
MIT License. Free to use, modify, and share.
