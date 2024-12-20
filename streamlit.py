
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load the pre-trained models
model_x = joblib.load('ridge_model_x.pkl')  # Ridge model for Product X
model_y = joblib.load('decision_tree_model_y.pkl')  # Decision Tree model for Product Y

# Load the dataset
file_path = 'Interview_dataset_ANALYTICS_EXECUTIVE.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Sheet1')

# Ensure 'datetime' column is in datetime format
df['datetime'] = pd.to_datetime(df['DateTime'])

# Extract year and month from the datetime column
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month_name()

df.rename(columns={
    'Product- X Vol (000s)': 'Product_X_Volume',
    'Product- Y  Vol(000s) ': 'Product_Y_Volume',
    'X  $/Unit': 'X_Price_Per_Unit',
    'Y $/unit': 'Y_Price_Per_Unit',
    'X conumers Mean Income': 'X_Consumers_Mean_Income',
    'Y Consumers Mean Income': 'Y_Consumers_Mean_Income',
    'Alternative Category % in the market': 'Alternative_Category_Percentage',
    'Counterfeit % in the market': 'Counterfeit_Percentage'
}, inplace=True)

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Forecasting Tool"

# Function to switch pages
def switch_page(page_name):
    st.session_state.page = page_name

# Navigation buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Forecasting Tool"):
    switch_page("Forecasting Tool")
if st.sidebar.button("Forecasting Tool - Manual Entry"):
    switch_page("Forecasting Tool - Manual Entry")
if st.sidebar.button("Exploratory Data Analysis (EDA)"):
    switch_page("EDA")
if st.sidebar.button("Year and Month-wise Analysis"):
    switch_page("Year and Month-wise Analysis")
if st.sidebar.button("Model Performance Analysis"):
    switch_page("Model Performance Analysis")

# Page Logic
if st.session_state.page == "Forecasting Tool":
    # FORECASTING TOOL PAGE
    latest_values = df.iloc[-1]
    st.title("Interactive Forecasting Tool")
    st.write("Adjust key drivers to dynamically update the next 12-month volume forecasts for Products X and Y.")

    # Input text boxes for user assumptions
    st.header("Input Assumptions for Forecast")
    counterfeit_change = float(st.text_input("Change in Counterfeit % (per month)", "0"))
    alternative_category_change = float(st.text_input("Change in Alternative Category % (per month)", "0"))
    price_change_x = float(st.text_input("Change in X Price Per Unit (monthly %)", "0"))
    price_change_y = float(st.text_input("Change in Y Price Per Unit (monthly %)", "0"))

    # Historical data points input
    history_length = int(st.text_input("Number of Historical Data Points", "100"))

    # Initialize forecast storage
    forecasted_volumes = {"Month": [], "Product X Volume": [], "Product Y Volume": []}

    # Generate forecasts for the next 12 months
    for step in range(1, 13):
        future_drivers_x = {
            "X_Price_Per_Unit": latest_values['X_Price_Per_Unit'] * (1 + (price_change_x / 100) * step),
            "X_Consumers_Mean_Income": latest_values['X_Consumers_Mean_Income'],
            "Alternative_Category_Percentage": latest_values['Alternative_Category_Percentage'] * (1 + (alternative_category_change / 100) * step),
            "Counterfeit_Percentage": latest_values['Counterfeit_Percentage'] * (1 + (counterfeit_change / 100) * step),
        }

        future_drivers_y = {
            "Alternative_Category_Percentage": latest_values['Alternative_Category_Percentage'] * (1 + (alternative_category_change / 100) * step),
            "Counterfeit_Percentage": latest_values['Counterfeit_Percentage'] * (1 + (counterfeit_change / 100) * step),
        }

        # Forecast volumes
        forecast_x = model_x.predict(np.array(list(future_drivers_x.values())).reshape(1, -1))[0]
        forecast_y = model_y.predict(np.array(list(future_drivers_y.values())).reshape(1, -1))[0]

        # Append forecasted values
        forecasted_volumes["Month"].append(f"Month {step}")
        forecasted_volumes["Product X Volume"].append(forecast_x)
        forecasted_volumes["Product Y Volume"].append(forecast_y)

    # Convert forecasts to DataFrame
    forecast_df = pd.DataFrame(forecasted_volumes)

    # Prepare historical data for visualization
    historical_x = df['Product_X_Volume'].tail(history_length).tolist()
    historical_y = df['Product_Y_Volume'].tail(history_length).tolist()
    historical_dates = df['datetime'].tail(history_length).tolist()

    # Generate forecasted dates dynamically
    last_date = historical_dates[-1]
    forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]

    # Combine historical and forecasted dates
    all_dates = historical_dates + forecast_dates

    # Include the last historical data point in forecasted data
    forecast_x = [historical_x[-1]] + forecast_df['Product X Volume'].tolist()
    forecast_y = [historical_y[-1]] + forecast_df['Product Y Volume'].tolist()

    # Ensure forecasted data aligns with dates
    forecast_dates = [last_date] + forecast_dates

    # Plot with matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(all_dates[:history_length], historical_x, label="Historical X", color="blue")
    plt.plot(all_dates[history_length - 1:], forecast_x, label="Forecasted X", color="lightblue", linestyle="dashed")
    plt.plot(all_dates[:history_length], historical_y, label="Historical Y", color="green")
    plt.plot(all_dates[history_length - 1:], forecast_y, label="Forecasted Y", color="lightgreen", linestyle="dashed")

    # Add labels, legend, and format x-axis
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title("Historical and Forecasted Volumes with Continuous Lines")
    plt.legend()
    plt.xticks(rotation=45)

    # Display plot in Streamlit
    st.pyplot(plt)
    
elif st.session_state.page == "EDA":
    # EDA PAGE
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Here are the visualizations and insights from the dataset for Products X and Y.")

    # Create separate dataframes for Product X and Product Y
    df1 = df[['Product_X_Volume', 'X_Price_Per_Unit', 'X_Consumers_Mean_Income', 'Alternative_Category_Percentage', 'Counterfeit_Percentage']]
    df2 = df[['Product_Y_Volume', 'Y_Price_Per_Unit', 'Y_Consumers_Mean_Income', 'Alternative_Category_Percentage', 'Counterfeit_Percentage']]

    # Dropdown menu for selecting the product
    product_choice = st.sidebar.selectbox("Select Product for Analysis", ["Product X", "Product Y"])

    if product_choice == "Product X":
        st.subheader("EDA for Product X")

        # Display the first few rows of df1
        st.write("Dataset for Product X:")
        st.dataframe(df1.head())

        st.subheader("Statistical Summary for Product X")
        st.write(df1.describe())

        # Correlation Heatmap for Product X
        st.subheader("Correlation Heatmap for Product X")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

        # Pairplot for Product X
        st.subheader("Pairplot for Product X")
        sns.pairplot(df1, diag_kind="kde")
        st.pyplot()

    elif product_choice == "Product Y":
        st.subheader("EDA for Product Y")

        # Display the first few rows of df2
        st.write("Dataset for Product Y:")
        st.dataframe(df2.head())

        st.subheader("Statistical Summary for Product Y")
        st.write(df2.describe())

        # Correlation Heatmap for Product Y
        st.subheader("Correlation Heatmap for Product Y")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df2.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

        # Pairplot for Product Y
        st.subheader("Pairplot for Product Y")
        sns.pairplot(df2, diag_kind="kde")
        st.pyplot()

# Page Logic for Year and Month-wise Analysis
elif st.session_state.page == "Year and Month-wise Analysis":
    st.title("Year and Month-wise Analysis")
    
    # Dropdowns for selecting type of analysis
    analysis_type = st.selectbox("Select Analysis Type", ["Month-wise Analysis", "Year-wise Analysis"])

    if analysis_type == "Month-wise Analysis":
        # Dropdown to select a specific month
        selected_month = st.selectbox("Select Month for Comparison", df['Month'].unique())
        
        # Filter data for the selected month
        month_data = df[df['Month'] == selected_month]

        month_data=month_data[['FY','Product_X_Volume', 'Product_Y_Volume']]
        
        # Group data by year for the selected month
        grouped_month_data = month_data.groupby('FY').mean().reset_index()
        
        st.subheader(f"Average Metrics for {selected_month} Across Years")
        st.dataframe(grouped_month_data)
        
        # Basic Streamlit line chart for trends
        st.subheader(f"Visualization of {selected_month} Data Across Years")
        st.line_chart(data=grouped_month_data.set_index('FY')[['Product_X_Volume', 'Product_Y_Volume']])
        

    elif analysis_type == "Year-wise Analysis":
        # Dropdown to select two FYs for comparison
        year1 = st.selectbox("Select First FY", df['FY'].unique(), key="year1")
        year2 = st.selectbox("Select Second FY", df['FY'].unique(), key="year2")

        # Filter data for the selected years
        year1_data = df[df['FY'] == year1]
        year2_data = df[df['FY'] == year2]
        
        # Filter data for the required fields
        year1_data= year1_data[['Product_X_Volume','Product_Y_Volume']]
        year2_data= year2_data[['Product_X_Volume','Product_Y_Volume']]
        
        # Calculate mean metrics for each year
        year1_summary = year1_data.sum()
        year2_summary = year2_data.sum()

        # Display the summaries
        st.subheader(f"Summary Metrics for FY{year1} and FY{year2}")
        comparison_df = pd.DataFrame({
            'Metric': ['Product_X_Volume', 'Product_Y_Volume'],
            f'FY{year1}': [year1_summary['Product_X_Volume'], year1_summary['Product_Y_Volume']],
            f'FY{year2}': [year2_summary['Product_X_Volume'], year2_summary['Product_Y_Volume']]
        })
        st.dataframe(comparison_df)

        # Basic Streamlit bar chart with FY on x-axis
        st.subheader(f"Visualization of FY{year1} vs FY{year2}")
        fy_comparison = pd.DataFrame({
            'FY': [f'FY{year1}', f'FY{year2}'],
            'Product_X_Volume': [year1_summary['Product_X_Volume'], year2_summary['Product_X_Volume']],
            'Product_Y_Volume': [year1_summary['Product_Y_Volume'], year2_summary['Product_Y_Volume']]
        })

        # Plot bar chart
        st.bar_chart(fy_comparison.set_index('FY'))
        

# Page Logic for Forecasting Tool - Manual Entry
if st.session_state.page == "Forecasting Tool - Manual Entry":
    st.title("Forecasting Tool - Manual Entry")
    st.write("Manually enter or edit actual values for each month to forecast volumes for Products X and Y.")

    # Initialize the DataFrame with monthly inputs
    latest_values = df.iloc[-1]
    last_date = latest_values['datetime']

    # Create initial input DataFrame
    input_data = pd.DataFrame({
        "Datetime": [last_date + pd.DateOffset(months=i) for i in range(1, 13)],
        "X_Price_Per_Unit": [1.76939261, 1.77386188, 1.77705795, 1.77614843, 1.77687087,
        1.79257038, 1.79792375, 1.79763792, 1.79750149, 1.79639777,1.80003852, 1.85025737],
        "Y_Price_Per_Unit": [2.63241849, 2.64750791, 2.66673132, 2.67820439, 2.69007945,
        2.72134706, 2.74034968, 2.75294943, 2.76750176, 2.78083697,2.79676291, 2.90503565],
        "X_Consumers_Mean_Income": [latest_values['X_Consumers_Mean_Income']] * 12,
        "Y_Consumers_Mean_Income": [latest_values['Y_Consumers_Mean_Income']] * 12,
        "Alternative_Category_Percentage": [0.45335567, 0.46564854, 0.46207008, 0.46462206, 0.48198054,
        0.48748704, 0.48203387, 0.48359557, 0.48952876, 0.48499432,0.49086799, 0.48861747],
        "Counterfeit_Percentage": [0.36502111, 0.3611247 , 0.35668494, 0.35219111, 0.35606494,
        0.35756808, 0.36429877, 0.36633718, 0.37169728, 0.37228948,0.36723531, 0.35805375]
    })

    # Allow users to edit the DataFrame
    st.subheader("Edit Input Data for Forecasting")
    edited_data = st.data_editor(input_data, use_container_width=True)

    # Combine historical data
    historical_data = df[["datetime", "Product_X_Volume", "Product_Y_Volume"]].tail(10)
    historical_data.columns = ["Datetime", "Product X Volume", "Product Y Volume"]

    # Use the edited DataFrame for forecasting
    forecasted_volumes = {"Datetime": [], "Product X Volume": [], "Product Y Volume": []}

    for _, row in edited_data.iterrows():
        # Prepare drivers for forecasting based on user input
        future_drivers_x = {
            "X_Price_Per_Unit": row["X_Price_Per_Unit"],
            "X_Consumers_Mean_Income": row["X_Consumers_Mean_Income"],
            "Alternative_Category_Percentage": row["Alternative_Category_Percentage"],
            "Counterfeit_Percentage": row["Counterfeit_Percentage"],
        }

        future_drivers_y = {
            "Alternative_Category_Percentage": row["Alternative_Category_Percentage"],
            "Counterfeit_Percentage": row["Counterfeit_Percentage"],
        }

        # Forecast volumes for the current month
        forecast_x = model_x.predict(np.array(list(future_drivers_x.values())).reshape(1, -1))[0]
        forecast_y = model_y.predict(np.array(list(future_drivers_y.values())).reshape(1, -1))[0]

        # Append forecasted values
        forecasted_volumes["Datetime"].append(row["Datetime"])
        forecasted_volumes["Product X Volume"].append(forecast_x)
        forecasted_volumes["Product Y Volume"].append(forecast_y)

    # Convert forecasted data to DataFrame
    forecast_df = pd.DataFrame(forecasted_volumes)

    # Add the last row of historical data as the first row of the forecasted data
    last_historical_row = historical_data.iloc[-1]
    forecast_df = pd.concat([pd.DataFrame([last_historical_row]), forecast_df]).reset_index(drop=True)

    # Plot historical and forecasted data separately
    st.subheader("Visualization of Historical and Forecasted Volumes")
    plt.figure(figsize=(12, 6))

    # Plot historical data for Product X and Product Y
    plt.plot(historical_data["Datetime"], historical_data["Product X Volume"], label="Historical X", color="blue", linewidth=2)
    plt.plot(historical_data["Datetime"], historical_data["Product Y Volume"], label="Historical Y", color="green", linewidth=2)

    # Plot forecasted data with dotted lines
    plt.plot(forecast_df["Datetime"], forecast_df["Product X Volume"], label="Forecasted X", color="blue", linestyle="dotted", linewidth=2)
    plt.plot(forecast_df["Datetime"], forecast_df["Product Y Volume"], label="Forecasted Y", color="green", linestyle="dotted", linewidth=2)

    # Add labels, legend, and format x-axis
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title("Historical and Forecasted Volumes with Continuous Lines")
    plt.legend()
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Display the forecasted DataFrame
    st.subheader("Forecasted Volumes")
    st.dataframe(forecast_df)


elif st.session_state.page == "Model Performance Analysis":
    st.title("Model Performance Analysis")
    
    
    image1_url = "result1.png"
    image2_url = "result2.png"
    vif1='vif1.png'
    vif2='vif2.png'
    
    # Display the image in the app
    st.image(image1_url, caption="Test Dataset Performance", use_column_width=True)
    st.image(image2_url, caption="Test Dataset Performance", use_column_width=True)

    st.subheader("VIF Test:")
    st.image(vif1, caption="With all features", use_column_width=True)
    st.image(vif2, caption="With only selected features", use_column_width=True)    


    # Provide interpretation of metrics and VIF
    st.subheader("Interpretation")
    st.write("""
    - **RMSE**: Lower values indicate better fit. It shows the average error magnitude in predicted values.
    - **MAPE**: A measure of prediction accuracy in percentage. Lower values are better.
    - **R²**: Higher values (close to 1) indicate a better model fit.
    - **VIF**: High VIF values (>10) indicate multicollinearity and potential overfitting in the model. Features with high VIF may need to be removed or transformed.
    """)

    st.subheader("Bias-Variance Tradeoff")
    st.write("""
    - Bias: The models exhibit very high 𝑅2  values (close to 1) on both the training and testing sets, indicating that the models are making highly accurate predictions. This suggests low bias.
    - Variance: The difference between the training and testing RMSE/metrics is relatively small, meaning the models are generalizing well and not overly sensitive to the training data. This suggests low variance.
    """)

