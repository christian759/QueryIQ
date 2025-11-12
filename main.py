import pandas as pd
import streamlit as st

# Load data
df = pd.read_csv("forex_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Streamlit UI
st.title("📈 Forex Price Dashboard (EUR/USD)")
st.subheader("Simulated Data Visualization")

# Display dataframe
st.dataframe(df)

# Plot Close Price
st.line_chart(df.set_index("timestamp")["close"])

# Add more metrics
st.area_chart(df.set_index("timestamp")[["open", "high", "low", "close"]])

# Show summary stats
st.write("### Summary Statistics")
st.write(df.describe())
