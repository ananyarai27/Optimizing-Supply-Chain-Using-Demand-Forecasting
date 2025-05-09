import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

st.set_page_config(page_title="Supply Chain Forecasting", layout="wide")
st.title("📦 Supply Chain Demand Forecasting Dashboard")

uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    try:
        df['week'] = pd.to_datetime(df['week'], format='%y/%m/%d')
    except:
        st.error("❌ Could not parse 'week' column — expected format: YY/MM/DD.")
        st.stop()

    if 'units_sold' not in df.columns:
        st.error("❌ 'units_sold' column missing.")
        st.stop()

    # --------- Feature 1: SKU & Store Filter ----------
    st.sidebar.subheader("🔍 Filter Options")
    store_options = ['All'] + sorted(df['store_id'].dropna().unique().tolist())
    sku_options = ['All'] + sorted(df['sku_id'].dropna().unique().tolist())

    selected_store = st.sidebar.selectbox("Select Store", store_options)
    selected_sku = st.sidebar.selectbox("Select SKU", sku_options)

    filtered_df = df.copy()
    if selected_store != 'All':
        filtered_df = filtered_df[filtered_df['store_id'] == selected_store]
    if selected_sku != 'All':
        filtered_df = filtered_df[filtered_df['sku_id'] == selected_sku]

    # --------- Feature 2: Date Range Picker ----------
    min_date = filtered_df['week'].min()
    max_date = filtered_df['week'].max()
    date_range = st.sidebar.date_input("📆 Select Date Range", [min_date, max_date],
                                       min_value=min_date, max_value=max_date)
    filtered_df = filtered_df[(filtered_df['week'] >= pd.to_datetime(date_range[0])) &
                              (filtered_df['week'] <= pd.to_datetime(date_range[1]))]

    # Preview
    st.subheader("🔍 Filtered Dataset Preview")
    st.write(f"**Rows:** {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(10))

    # Aggregate
    weekly_demand = filtered_df.groupby('week')['units_sold'].sum().reset_index()
    weekly_demand['forecast'] = weekly_demand['units_sold'].rolling(window=4, min_periods=1).mean().shift(1)

    # Forecast Accuracy
    y_true = weekly_demand['units_sold'].iloc[4:]
    y_pred = weekly_demand['forecast'].iloc[4:]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE", f"{mae:,.0f}")
    col2.metric("📉 RMSE", f"{rmse:,.0f}")
    col3.metric("📉 MAPE (%)", f"{mape:.2f}%")

    # --------- Line Chart: Actual vs Forecast ----------
    st.subheader("📈 Weekly Demand: Actual vs Forecast (with Promotion Highlights)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_demand['week'], weekly_demand['units_sold'], label='Actual', marker='o')
    ax.plot(weekly_demand['week'], weekly_demand['forecast'], label='Forecast', linestyle='--', marker='x')

    # --------- Feature 3: Promotion Highlights ----------
    if 'is_featured_sku' in filtered_df.columns or 'is_display_sku' in filtered_df.columns:
        promo_weeks = filtered_df[filtered_df['is_featured_sku'] == 1]['week'].unique()
        for pw in promo_weeks:
            ax.axvline(pw, color='orange', linestyle=':', alpha=0.3, label='Featured SKU Week')
        display_weeks = filtered_df[filtered_df['is_display_sku'] == 1]['week'].unique()
        for dw in display_weeks:
            ax.axvline(dw, color='purple', linestyle=':', alpha=0.3, label='Display SKU Week')

    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    ax.set_title("Actual vs Forecasted Demand")
    ax.legend()
    st.pyplot(fig)

    # --------- Enhanced Weekly Sales Volume Chart ----------
    st.subheader("📊 Weekly Sales Volume (Enhanced)")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis((weekly_demand['units_sold'] - weekly_demand['units_sold'].min()) /
                            (weekly_demand['units_sold'].max() - weekly_demand['units_sold'].min()))

    bars = ax2.bar(weekly_demand['week'], weekly_demand['units_sold'], color=colors, edgecolor='black')

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    # Highlight peak
    peak_idx = weekly_demand['units_sold'].idxmax()
    peak_week = weekly_demand.loc[peak_idx, 'week']
    ax2.axvline(peak_week, color='red', linestyle='--', alpha=0.6, label='Peak Week')

    ax2.set_xlabel("Week", fontsize=11)
    ax2.set_ylabel("Units Sold", fontsize=11)
    ax2.set_title("Total Weekly Demand (Bar Intensity + Peak Highlighted)", fontsize=13)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.legend()
    st.pyplot(fig2)

    # --------- Downloads Section ----------
    st.subheader("⬇️ Download Forecast Data")

    # CSV Download
    csv = weekly_demand.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="forecast_output.csv", mime="text/csv")

    # PDF Download
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')
    pdf_buffer.seek(0)
    st.download_button("Download Charts as PDF", pdf_buffer, file_name="forecast_charts.pdf", mime="application/pdf")

else:
    st.info("👈 Upload a CSV file with 'week' and 'units_sold' columns to begin.")
