import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Custom CSS to apply a darker theme with lighter plot backgrounds
st.set_page_config(layout="wide")
custom_css = """
<style>
    body {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stApp {
        background-color: #2d2d2d;
    }
    .css-18e3th9 {
        background-color: #f7f7f7;
        border-radius: 10px;
    }
    .css-1d391kg p {
        color: #ffffff;
    }
    .css-1v3fvcr {
        color: #cccccc;
    }
</style>
"""

# Custom CSS for wider plot
st.markdown(
    """
    <style>
    .element-container {
        width: 75% !important;
        margin: auto;
    }
    .plot-container {
        width: 800px !important;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load CSV file for securities
holdings_file = "cleaned_combined_holdings_with_industry.csv"
df_holdings = pd.read_csv(holdings_file)

# Load CSV files for MWR (USD and CAD)
usd_mwr_file = "returns_data_usd.csv"
cad_mwr_file = "returns_data_cad.csv"
usd_data = pd.read_csv(usd_mwr_file)
cad_data = pd.read_csv(cad_mwr_file)

# Function to get the last day of the month
def last_day_of_month(date):
    if pd.isnull(date):
        return None
    date_obj = pd.to_datetime(date)
    next_month = date_obj.replace(day=28) + timedelta(days=4)
    return (next_month - timedelta(days=next_month.day)).date()

# Process holdings data
df_holdings["Date"] = df_holdings["Date"].apply(last_day_of_month)
df_holdings["Date"] = pd.to_datetime(df_holdings["Date"])
df_holdings["Market Value"] = pd.to_numeric(df_holdings["Market Value"], errors="coerce").fillna(0)
df_holdings["Book Cost"] = pd.to_numeric(df_holdings["Book Cost"], errors="coerce").fillna(0)
df_holdings["CADUSD Conversion Rate"] = pd.to_numeric(df_holdings["CADUSD Conversion Rate"], errors="coerce").fillna(1)
df_holdings["USDCAD Conversion Rate"] = pd.to_numeric(df_holdings["USDCAD Conversion Rate"], errors="coerce").fillna(1)

# Remove rows where Market Value is 0
df_holdings = df_holdings[df_holdings["Market Value"] > 0]

# Sidebar currency selection
currency = st.sidebar.radio(
    "Select Currency:",
    ["USD", "CAD"],
    index=0  # Default to "USD"
)

page = st.sidebar.radio(
    "Select Page:",
    ["Cumulative Return","Evolution of Securities"],
    index=0  # Default to "Cumulative Return"
)

if page == "Cumulative Return":
    slicer = st.sidebar.selectbox(
        "Select Time Range:",
        ["3 Months", "6 Months", "Year to Date", "12 Months", "18 Months", "3 Years", "5 Years", "Since Inception"],
        index=2  # Default to "Year to Date"
    )

# Adjust based on currency
if currency == "USD":
    df_holdings["Weighted Market Value"] = df_holdings["Market Value"] * df_holdings["CADUSD Conversion Rate"]
    df_holdings["Weighted Book Cost"] = df_holdings["Book Cost"] * df_holdings["CADUSD Conversion Rate"]
    y_label = "USD Value"
    combined_data = usd_data
    # Convert to the last day of the month
    combined_data["Date"] = pd.to_datetime(combined_data["Date"])
    combined_data["Date"] = combined_data["Date"] + pd.offsets.MonthEnd(0)
    
    # Group and pivot data for Market Value and Book Cost
    grouped_holdings = df_holdings.groupby(["Date", "Ticker"])["Weighted Market Value"].sum().reset_index()
    grouped_book_cost = df_holdings.groupby(["Date", "Ticker"])["Weighted Book Cost"].sum().reset_index()

    pivot_holdings = grouped_holdings.pivot(index="Date", columns="Ticker", values="Weighted Market Value").fillna(0)
    pivot_book_cost = grouped_book_cost.pivot(index="Date", columns="Ticker", values="Weighted Book Cost").fillna(0)

    pivot_holdings = pivot_holdings.reset_index()
    pivot_book_cost = pivot_book_cost.reset_index()

    # Calculate percentages manually
    total_value = pivot_holdings.drop(columns=["Date"]).sum(axis=1)
    pivot_percentages = pivot_holdings.copy()

    for col in pivot_percentages.columns[1:]:
        pivot_percentages[col] = (pivot_percentages[col] / total_value) * 100

    # Calculate % Gain for each ticker
    pivot_gain = pivot_holdings.copy()
    for col in pivot_gain.columns[1:]:
        pivot_gain[col] = ((pivot_holdings[col] / pivot_book_cost[col]) - 1) * 100

    # Melt data for plotting
    pivot_melted = pivot_holdings.melt(id_vars=["Date"], var_name="Ticker", value_name="Value")
    pivot_percent_melted = pivot_percentages.melt(id_vars=["Date"], var_name="Ticker", value_name="Percentage")
    pivot_gain_melted = pivot_gain.melt(id_vars=["Date"], var_name="Ticker", value_name="% Gain")

    combined_plot_data = pd.merge(pivot_melted, pivot_percent_melted, on=["Date", "Ticker"])
    combined_plot_data = pd.merge(combined_plot_data, pivot_gain_melted, on=["Date", "Ticker"])

else:
    df_holdings["Weighted Market Value CAD"] = df_holdings["Market Value"] * df_holdings["USDCAD Conversion Rate"]
    df_holdings["Weighted Book Cost CAD"] = df_holdings["Book Cost"] * df_holdings["USDCAD Conversion Rate"]
    y_label = "CAD Value"
    combined_data = cad_data
    # Convert to the last day of the month
    combined_data["Date"] = pd.to_datetime(combined_data["Date"])
    combined_data["Date"] = combined_data["Date"] + pd.offsets.MonthEnd(0)

    # Group and pivot data for Market Value and Book Cost
    grouped_holdings = df_holdings.groupby(["Date", "Ticker"])["Weighted Market Value CAD"].sum().reset_index()
    grouped_book_cost = df_holdings.groupby(["Date", "Ticker"])["Weighted Book Cost CAD"].sum().reset_index()

    pivot_holdings = grouped_holdings.pivot(index="Date", columns="Ticker", values="Weighted Market Value CAD").fillna(0)
    pivot_book_cost = grouped_book_cost.pivot(index="Date", columns="Ticker", values="Weighted Book Cost CAD").fillna(0)

    pivot_holdings = pivot_holdings.reset_index()
    pivot_book_cost = pivot_book_cost.reset_index()

    # Calculate percentages manually
    total_value = pivot_holdings.drop(columns=["Date"]).sum(axis=1)
    pivot_percentages = pivot_holdings.copy()

    for col in pivot_percentages.columns[1:]:
        pivot_percentages[col] = (pivot_percentages[col] / total_value) * 100

    # Calculate % Gain for each ticker
    pivot_gain = pivot_holdings.copy()
    for col in pivot_gain.columns[1:]:
        pivot_gain[col] = ((pivot_holdings[col] / pivot_book_cost[col]) - 1) * 100

    # Melt data for plotting
    pivot_melted = pivot_holdings.melt(id_vars=["Date"], var_name="Ticker", value_name="Value")
    pivot_percent_melted = pivot_percentages.melt(id_vars=["Date"], var_name="Ticker", value_name="Percentage")
    pivot_gain_melted = pivot_gain.melt(id_vars=["Date"], var_name="Ticker", value_name="% Gain")

    combined_plot_data = pd.merge(pivot_melted, pivot_percent_melted, on=["Date", "Ticker"])
    combined_plot_data = pd.merge(combined_plot_data, pivot_gain_melted, on=["Date", "Ticker"])

# Filter out zero-value entries before plotting
combined_plot_data = combined_plot_data[combined_plot_data["Value"] > 0]



# Plot for Securities Evolution
if page == "Evolution of Securities":
    st.title("Evolution of Securities")

    # Pass custom hover data (Ticker, Value, Percentage, % Gain)
    fig = px.area(
        combined_plot_data,
        x="Date",
        y="Value",
        color="Ticker",
        labels={"Value": y_label, "Percentage": "Percentage (%)"},
        custom_data=["Ticker", "Value", "Percentage", "% Gain"]  # Pass data to hover
    )
    
    # Update hover to show full details including % Gain
    fig.update_traces(
        line=dict(width=0),  # No line, for smoother fills
        hoveron='points+fills',  # Hover over area fills and points
        hovertemplate="<b>%{customdata[0]}</b><br>"  # Ticker
                      "Value: %{customdata[1]:,.2f}<br>"  # Value formatted with commas
                      "Percentage: %{customdata[2]:.2f}%<br>"  # Percentage
                      "Gain: %{customdata[3]:.2f}%<extra></extra>"  # % Gain formatted
    )

    # Adjust layout for better visuals
    fig.update_layout(
        plot_bgcolor="#f7f7f7",
        paper_bgcolor="#2d2d2d"
    )

    # Display the plot
    st.plotly_chart(fig)

elif page == "Cumulative Return":
    st.title("Cumulative Return Over Time")
    combined_data["MonthYear"] = pd.to_datetime(combined_data["MonthYear"])#, format="%m-%Y")
    combined_data = combined_data.sort_values(by="MonthYear")
    combined_data["Cumulative MWR"] = (1 + combined_data["Monthly MWR"]).cumprod() - 1
    #combined_data["Cumulative SPX Adjusted Return"] = (1 + combined_data["SPX Adjusted Return"]).cumprod() - 1

    combined_data["Cumulative MWR"] -= combined_data["Cumulative MWR"].iloc[0]
    #combined_data["Cumulative SPX Adjusted Return"] -= combined_data["Cumulative SPX Adjusted Return"].iloc[0]

    fig3 = go.Figure()
    fig2 = go.Figure()
    # Apply date slicer for filtering
    end_date = combined_data["MonthYear"].max()

    if slicer == "3 Months":
        start_date = end_date - pd.DateOffset(months=2)
    elif slicer == "6 Months":
        start_date = end_date - pd.DateOffset(months=5)
    elif slicer == "Year to Date":
        # Start from the beginning of the current year
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif slicer == "12 Months":
        start_date = end_date - pd.DateOffset(months=11)
    elif slicer == "18 Months":
        start_date = end_date - pd.DateOffset(months=17)
    elif slicer == "3 Years":
        start_date = end_date - pd.DateOffset(years=3)
    elif slicer == "5 Years":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = combined_data["MonthYear"].min()

    # Filter the data based on slicer
    filtered_data = combined_data[(combined_data["MonthYear"] >= start_date) & (combined_data["MonthYear"] <= end_date)].copy()
    # Convert MonthYear (MM-YYYY) to the last day of the month
    filtered_data["MonthYear"] = pd.to_datetime(filtered_data["MonthYear"], format="%m-%Y") + pd.offsets.MonthEnd(0)


    # CAD-specific cumulative return logic with baseline row
    if currency == "USD":
        # Calculate cumulative returns as usual
        filtered_data["Cumulative MWR"] = (1 + filtered_data["Monthly MWR"]).cumprod() - 1
        filtered_data["Cumulative SPX Adjusted Return"] = (1 + filtered_data["SPX Adjusted Return"]).cumprod() - 1
    else:
        filtered_data["Cumulative MWR"] = (1 + filtered_data["Monthly MWR"]).cumprod() - 1
        filtered_data["Cumulative TSX Adjusted Return"] = (1 + filtered_data["TSX Adjusted Return"]).cumprod() - 1

    # Insert a baseline row to start at 0 one month before the first date
    baseline_date = filtered_data["MonthYear"].min() - pd.DateOffset(months=1)
    baseline_date = baseline_date + pd.offsets.MonthEnd(0)  # Shift to the last day of the previous month

    baseline_row = {
        "MonthYear": baseline_date,
        "Monthly MWR": 0,
        "SPX Adjusted Return": 0 if currency == "USD" else None,
        "TSX Adjusted Return": 0 if currency == "CAD" else None,
        "Cumulative MWR": 0,
        "Cumulative SPX Adjusted Return": 0 if currency == "USD" else None,
        "Cumulative TSX Adjusted Return": 0 if currency == "CAD" else None
    }

    # Add baseline to filtered data
    filtered_data = pd.concat([pd.DataFrame([baseline_row]), filtered_data]).reset_index(drop=True)


    # Recalculate cumulative returns after inserting the baseline
    if currency == "USD":
        filtered_data["Cumulative MWR"] = (1 + filtered_data["Monthly MWR"]).cumprod() - 1
        filtered_data["Cumulative SPX Adjusted Return"] = (1 + filtered_data["SPX Adjusted Return"]).cumprod() - 1
    else:
        filtered_data["Cumulative MWR"] = (1 + filtered_data["Monthly MWR"]).cumprod() - 1
        filtered_data["Cumulative TSX Adjusted Return"] = (1 + filtered_data["TSX Adjusted Return"]).cumprod() - 1

    # Plot cumulative returns
    fig3.add_trace(go.Scatter(
        x=filtered_data["MonthYear"],
        y=filtered_data["Cumulative MWR"] * 100,
        mode='lines',
        line=dict(color='red', width=2),
        name="Cumulative MWR"
    ))

    fig3.add_trace(go.Scatter(
        x=filtered_data["MonthYear"],
        y=filtered_data["Cumulative SPX Adjusted Return"] * 100 if currency == "USD" else filtered_data["Cumulative TSX Adjusted Return"] * 100,
        mode='lines',
        line=dict(color='black', width=2),
        name="Market Return" +" (SPX)" if currency == "USD" else "Market Return" +" (TSX)"
    ))

    fig3.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Cumulative Return (%)",
        plot_bgcolor="#f7f7f7",
        paper_bgcolor="#2d2d2d",
        width=2000,  
        height=500,  
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        autosize=True,
        xaxis=dict(domain=[0, 0.75]),  # Cumulative return plot takes 65% of width
        xaxis2=dict(domain=[0.05, 1])   # Bar chart starts at 70%, creating separation
    )

    
    combined_data["MonthYear"] = pd.to_datetime(combined_data["MonthYear"], format="%m-%Y")
    if currency == "USD":
        combined_data["Adjusted Return"] = combined_data["SPX Adjusted Return"]
    else:
        combined_data["Adjusted Return"] = combined_data["TSX Adjusted Return"]
    # Split data into positive and negative returns
    positive_data = filtered_data[filtered_data["Monthly MWR"] > 0]
    negative_data = filtered_data[filtered_data["Monthly MWR"] <= 0]

    fig2 = go.Figure()

    # Positive Bars (Green)
    fig2.add_trace(go.Bar(
        x=positive_data["MonthYear"].iloc[1:],
        y=positive_data["Monthly MWR"].iloc[1:] * 100,
        marker_color="green",
        opacity=1,
        name="Positive MWR"
    ))

    # Negative Bars (Red)
    fig2.add_trace(go.Bar(
        x=negative_data["MonthYear"].iloc[1:],
        y=negative_data["Monthly MWR"].iloc[1:] * 100,
        marker_color="red",
        opacity=1,
        name="Negative MWR"
    ))

    # Market Return Scatter
    fig2.add_trace(go.Scatter(
        x=filtered_data["MonthYear"].iloc[1:],
        y=filtered_data["SPX Adjusted Return"].iloc[1:] * 100 if currency == "USD" else filtered_data["TSX Adjusted Return"].iloc[1:] * 100,
        mode='markers',
        marker=dict(color='black', size=4),
        name="Market Return"
    ))

    # Layout
    fig2.update_layout(
        plot_bgcolor="#f7f7f7",
        paper_bgcolor="#2d2d2d",
        width=2000,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        autosize=True
    )



    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    def calculate_statistics(returns, market_returns):
        # Ensure cumulative return starts at 0 before the first return
        cumulative_returns = (1 + returns).cumprod() - 1
        cumulative_return = cumulative_returns.iloc[-1]  # Terminal value at the end of the period

        # Volatility (Annualized)
        volatility = returns.std() * (12 ** 0.5)
        
        # Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * (12 ** 0.5) if returns.std() != 0 else 0

        # Rolling Max for Drawdown
        rolling_max = (1 + returns).cumprod().cummax()
        drawdown = (1 + returns).cumprod() / rolling_max - 1
        max_drawdown = drawdown.min()

        # Average Annualized Return
        avg_return = returns.mean() * 12  

        # Beta Calculation
        if len(market_returns) > 1 and returns.cov(market_returns) > 0:
            beta = returns.cov(market_returns) / market_returns.var()
        else:
            beta = np.nan

        # Sortino Ratio (Downside Risk)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * (12 ** 0.5)
        sortino_ratio = returns.mean() / downside_volatility * (12 ** 0.5) if downside_volatility != 0 else 0

        # Skewness of Returns
        skewness = returns.skew()

        # Return statistics as dictionary
        return {
            "Cumulative Return": cumulative_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown,
            "Average Return (Annualized)": avg_return,
            "Beta": beta,
            "Skewness": skewness
        }


    fund_stats = calculate_statistics(filtered_data["Monthly MWR"].iloc[1:], 
                                      filtered_data["SPX Adjusted Return"].iloc[1:] if currency == "USD" 
                                      else filtered_data["TSX Adjusted Return"].iloc[1:])

    market_stats = calculate_statistics(filtered_data["SPX Adjusted Return"].iloc[1:], 
                                        filtered_data["SPX Adjusted Return"].iloc[1:]) if currency == "USD" else \
                   calculate_statistics(filtered_data["TSX Adjusted Return"].iloc[1:], 
                                        filtered_data["TSX Adjusted Return"].iloc[1:])

    stats_df = pd.DataFrame({
        "Fund": fund_stats,
        "Market": market_stats
    })

    if currency == "USD":
        months_beat_market = (filtered_data["Monthly MWR"].iloc[1:] > filtered_data["SPX Adjusted Return"].iloc[1:]).sum()
    else:
        months_beat_market = (filtered_data["Monthly MWR"].iloc[1:] > filtered_data["TSX Adjusted Return"].iloc[1:]).sum()

    total_months = len(filtered_data)
    percentage_beat_market = (months_beat_market / total_months) * 100 if total_months > 0 else 0

    col1, col2, col3 = st.columns([1.25, 2, 2])  # Three columns (1 for overview, 2 for stats, 2 for pie chart)

    with col1:
        st.write("### Performance Overview")
        st.markdown(f"**Percentage of Months Beating Market:** {percentage_beat_market:.1f}%")

    with col2:
        st.write("### Portfolio Statistics for Selected Period")
        st.dataframe(stats_df.style.format({
            "Fund": "{:.2%}",
            "Market": "{:.2%}"
        }))

    with col3:
        st.write("### Ticker Allocation")
        
        # Get the latest allocation (last row of pivot_percentages)
        latest_allocation = pivot_percentages.iloc[-1].drop("Date")
        
        # Prepare DataFrame for Plotly
        allocation_df = latest_allocation.reset_index()
        allocation_df.columns = ["Ticker", "Percentage"]

    # Filter out tickers with 0% allocation for plotting
    allocation_df = allocation_df[allocation_df["Percentage"] >= 1 ]

    # Plot Donut Chart
    fig_pie = px.pie(
        allocation_df,
        names="Ticker",
        values="Percentage",
        hole=0.4#,  # Donut style
        #title=f"Allocation as of {pivot_percentages['Date'].max().strftime('%Y-%m-%d')}"
    )

    # Make 0% tickers invisible in legend
    for trace in fig_pie.data:
        if trace.labels[trace.values == 0].any():
            trace.textfont.color = "rgba(0,0,0,0)"  # Fully transparent
            trace.marker.line.color = "rgba(0,0,0,0)"  # Hide outline if needed

    # Update layout
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(
        width=500,
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig_pie, use_container_width=True)


