import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import google.generativeai as genai
import warnings
import os

warnings.filterwarnings('ignore')


def clean_data(df):
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    
    df.set_index('createdAt', inplace=True)
    
    df.sort_index(inplace=True)
    
    df = df[['flow1', 'turbidity', 'ph', 'tds']]

    df = df.loc['2025-02-27':'2025-06-01']
    df_daily = df.resample('D').mean()
    
    df_daily = df_daily.fillna(method='ffill')
    
    return df_daily

def create_forecast_prompt(df, col_name, forecast_days):
    historical_data = df[col_name].tolist()
    
    prompt = f"""
    You are a water quality analysis expert with specialized expertise in time series forecasting.
    I have historical data for the {col_name} parameter as follows (daily data):
    {historical_data}
    
    Please predict the next {forecast_days} days based on these historical patterns.
    
    Expected output format:
    - Prediction day 1: [value]
    - Prediction day 2: [value]
    - ...
    - Prediction day {forecast_days}: [value]
    
    Provide only the list of predicted values in the above format, without additional explanations.
    Ensure predicted values are consistent with historical data patterns.
    """
    return prompt

def create_insight_prompt(df, col_name, predictions, treatment_type):
    prompt = f"""
    You are a water quality analysis expert. Here is historical data and predictions for the {col_name} parameter ({treatment_type} treatment):
    
    Historical Data:
    {df[col_name].tail().tolist()}
    
    Prediction Results (next {len(predictions)} days):
    {predictions}
    
    Provide professional insights about:
    1. The difference in {col_name} parameter trends before and after treatment
    
    Output format:
    - **Trend Comparison**: [trend difference analysis]
    """
    return prompt

def create_comparison_insight_prompt(before_data, after_data, col_name):
    prompt = f"""
    You are a water quality analysis expert. Here is a comparison of {col_name} parameter data before and after treatment:
    
    Pre-Treatment Data (last 5 entries):
    {before_data[col_name].tail().tolist()}
    
    Post-Treatment Data (last 5 entries):
    {after_data[col_name].tail().tolist()}
    
    Provide professional insights about:
    1. The difference in predicted trends for {col_name} parameter between before (panel A) and after (panel E) treatment
    
    Output format:
    - **Prediction Trend Comparison**: [trend difference analysis]
    """
    return prompt

def get_gemini_response(prompt):
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None

def parse_prediction_output(output, forecast_days):
    predictions = []
    for line in output.split('\n'):
        if 'Prediction day' in line and ':' in line:
            try:
                pred = float(line.split(':')[1].strip())
                predictions.append(pred)
            except:
                continue
    return predictions[:forecast_days] if len(predictions) >= forecast_days else None

def plot_comparison(before_df, after_df, col_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    before_df[col_name].plot(
        ax=ax,
        label='Pre-Treatment',
        color='blue',
        alpha=0.7,
        marker='o'
    )
    
    after_df[col_name].plot(
        ax=ax,
        label='Post-Treatment',
        color='green',
        alpha=0.7,
        marker='s'
    )
    
    ax.set_title(f'{col_name.upper()} Comparison Before and After Treatment')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_forecast_comparison(before_df, after_df, before_pred, after_pred, col_name, forecast_days):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    before_df[col_name].plot(
        ax=ax,
        label='Pre-Treatment (Historical)',
        color='blue',
        alpha=0.5,
        marker='o'
    )
    
    after_df[col_name].plot(
        ax=ax,
        label='Post-Treatment (Historical)',
        color='green',
        alpha=0.5,
        marker='s'
    )
    
    last_before_date = before_df.index[-1]
    future_before_dates = [last_before_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    last_after_date = after_df.index[-1]
    future_after_dates = [last_after_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    ax.plot(
        future_before_dates,
        before_pred,
        label='Pre-Treatment (Prediction)',
        color='orange',
        linestyle='--',
        marker='o'
    )
    
    ax.plot(
        future_after_dates,
        after_pred,
        label='Post-Treatment (Prediction)',
        color='red',
        linestyle='--',
        marker='s'
    )
    
    ax.axvline(x=last_before_date, color='blue', linestyle=':', alpha=0.7)
    ax.axvline(x=last_after_date, color='green', linestyle=':', alpha=0.7)
    
    ax.set_title(f'{col_name.upper()} Forecast Comparison ({forecast_days} Days)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def main():
    st.title('Water Quality Forecasting Dashboard')
    st.write("""
    Daily water quality prediction application with before-and-after treatment comparison.
    Displays forecasts for all parameters (flow1, turbidity, pH, TDS).
    """)
    
    with st.sidebar:
        st.header('Settings')
        st.subheader("Upload Dataset")
        before_file = st.file_uploader("Upload CSV dataset (Pre-Treatment)", type=['csv'])
        after_file = st.file_uploader("Upload CSV dataset (Post-Treatment)", type=['csv'])
        
        st.subheader("Gemini API Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        forecast_days = st.slider(
            'Number of forecast days',
            min_value=1, max_value=30, value=7
        )
    
    if before_file is not None and after_file is not None:
        # Load dan bersihkan data
        before_df = pd.read_csv(before_file)
        after_df = pd.read_csv(after_file)
        
        before_clean = clean_data(before_df)
        after_clean = clean_data(after_df)
        
        st.subheader('Daily Water Quality Data')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pre-Treatment**")
            st.dataframe(before_clean.tail(), use_container_width=True)
        with col2:
            st.markdown("**Post-Treatment**")
            st.dataframe(after_clean.tail(), use_container_width=True)
        
        st.subheader('Before vs After Treatment Comparison')
        
        for col in ['flow1', 'turbidity', 'ph', 'tds']:
            # Plot komparatif
            fig = plot_comparison(before_clean, after_clean, col)
            st.pyplot(fig)
            
            insight_prompt = create_comparison_insight_prompt(before_clean, after_clean, col)
            insight_output = get_gemini_response(insight_prompt)
            
            if insight_output:
                with st.expander(f"{col.upper()} Comparison Insights"):
                    st.markdown(insight_output)
            
            st.markdown("---")
        
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = {
                'before': {},
                'after': {}
            }
        
        if st.button('Generate Forecasts for All Parameters'):
            if 'gemini_api_key' not in st.session_state:
                st.error("Please enter Gemini API Key first")
                st.stop()
                return
                
            with st.spinner('Generating predictions and analysis...'):
                try:
                    results_container = st.container()
                    
                    for col in ['flow1', 'turbidity', 'ph', 'tds']:
                        with results_container:
                            st.subheader(f'{col.upper()} Parameter Forecast')
                            
                            before_prompt = create_forecast_prompt(before_clean, col, forecast_days)
                            before_output = get_gemini_response(before_prompt)
                            before_pred = parse_prediction_output(before_output, forecast_days)
                            
                            after_prompt = create_forecast_prompt(after_clean, col, forecast_days)
                            after_output = get_gemini_response(after_prompt)
                            after_pred = parse_prediction_output(after_output, forecast_days)
                            
                            if not before_pred or not after_pred:
                                st.error(f"Failed to process predictions for {col}")
                                continue
                            
                            fig = plot_forecast_comparison(
                                before_clean, after_clean, 
                                before_pred, after_pred,
                                col, forecast_days
                            )
                            st.pyplot(fig)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Pre-Treatment Forecast**")
                                last_before_date = before_clean.index[-1]
                                future_before_dates = [last_before_date + timedelta(days=i+1) for i in range(forecast_days)]
                                before_forecast_df = pd.DataFrame({
                                    'Date': future_before_dates,
                                    'Prediction': before_pred
                                }).set_index('Date')
                                st.dataframe(before_forecast_df.style.format({'Prediction': '{:.4f}'}), 
                                            use_container_width=True)
                            
                            with col2:
                                st.markdown("**Post-Treatment Forecast**")
                                last_after_date = after_clean.index[-1]
                                future_after_dates = [last_after_date + timedelta(days=i+1) for i in range(forecast_days)]
                                after_forecast_df = pd.DataFrame({
                                    'Date': future_after_dates,
                                    'Prediction': after_pred
                                }).set_index('Date')
                                st.dataframe(after_forecast_df.style.format({'Prediction': '{:.4f}'}), 
                                            use_container_width=True)
                            
                            before_insight_prompt = create_insight_prompt(before_clean, col, before_pred, "pre")
                            before_insight_output = get_gemini_response(before_insight_prompt)
                            
                            after_insight_prompt = create_insight_prompt(after_clean, col, after_pred, "post")
                            after_insight_output = get_gemini_response(after_insight_prompt)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if before_insight_output:
                                    with st.expander(f"Pre-Treatment Insights ({col.upper()})"):
                                        st.markdown(before_insight_output)
                            
                            with col2:
                                if after_insight_output:
                                    with st.expander(f"Post-Treatment Insights ({col.upper()})"):
                                        st.markdown(after_insight_output)
                            
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload pre-treatment and post-treatment CSV datasets to begin analysis.")

if __name__ == '__main__':
    main()