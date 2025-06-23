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

# Fungsi untuk membersihkan data
def clean_data(df):
    # Konversi kolom timestamp dan createdAt ke datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    
    # Set createdAt sebagai index
    df.set_index('createdAt', inplace=True)
    
    df.sort_index(inplace=True)
    
    # Hanya ambil kolom yang diperlukan
    df = df[['flow1', 'turbidity', 'ph', 'tds']]

    df = df.loc['2025-02-27':'2025-06-01']
    
    # Resample ke daily dan ambil mean
    df_daily = df.resample('D').mean()
    
    # Handle missing values dengan interpolasi
    df_daily = df_daily.fillna(method='ffill')
    
    return df_daily

# Fungsi untuk membuat prompt forecasting LLM
def create_forecast_prompt(df, col_name, forecast_days):
    # Gunakan semua data historis
    historical_data = df[col_name].tolist()
    
    prompt = f"""
    Anda adalah seorang ahli analisis kualitas air dengan keahlian khusus dalam time series forecasting.
    Saya memiliki data historis parameter {col_name} sebagai berikut (data harian):
    {historical_data}
    
    Buatlah prediksi untuk {forecast_days} hari ke depan berdasarkan pola data historis tersebut.
    
    Format output yang diharapkan:
    - Prediksi hari 1: [nilai]
    - Prediksi hari 2: [nilai]
    - ...
    - Prediksi hari {forecast_days}: [nilai]
    
    Berikan hanya daftar angka prediksi saja dalam format di atas, tanpa penjelasan tambahan.
    Pastikan nilai prediksi konsisten dengan pola data historis.
    """
    return prompt

# Fungsi untuk membuat prompt insight
def create_insight_prompt(df, col_name, predictions, treatment_type):
    prompt = f"""
    Anda adalah seorang ahli analisis kualitas air. Berikut adalah data historis dan prediksi untuk parameter {col_name} ({treatment_type} treatment):
    
    Data Historis (5 data terakhir):
    {df[col_name].tail().tolist()}
    
    Hasil Prediksi ({len(predictions)} hari ke depan):
    {predictions}
    
     Berikan insight profesional tentang:
    1. Perbedaan tren parameter {col_name} sebelum dan setelah treatment
    
    Format output:
    - **Perbandingan Tren Prediksi**: [analisis perbedaan tren]
    """
    return prompt

# Fungsi untuk membuat prompt insight komparatif
def create_comparison_insight_prompt(before_data, after_data, col_name):
    prompt = f"""
    Anda adalah seorang ahli analisis kualitas air. Berikut adalah perbandingan data parameter {col_name} sebelum dan setelah treatment:
    
    Data Sebelum Treatment (5 data terakhir):
    {before_data[col_name].tail().tolist()}
    
    Data Setelah Treatment (5 data terakhir):
    {after_data[col_name].tail().tolist()}
    
     Berikan insight profesional tentang:
    1. Perbedaan tren prediksi parameter {col_name} sebelum (panel A) dan setelah (panel E) treatment
    
    Format output:
    - **Perbandingan Tren Prediksi**: [analisis perbedaan tren]
    """
    return prompt

# Fungsi untuk mendapatkan prediksi dari Gemini
def get_gemini_response(prompt):
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error saat memanggil Gemini API: {str(e)}")
        return None

# Fungsi untuk parsing output prediksi
def parse_prediction_output(output, forecast_days):
    predictions = []
    for line in output.split('\n'):
        if 'Prediksi hari' in line and ':' in line:
            try:
                pred = float(line.split(':')[1].strip())
                predictions.append(pred)
            except:
                continue
    return predictions[:forecast_days] if len(predictions) >= forecast_days else None

# Fungsi untuk plot komparatif
def plot_comparison(before_df, after_df, col_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data sebelum treatment
    before_df[col_name].plot(
        ax=ax,
        label='Sebelum Treatment',
        color='blue',
        alpha=0.7,
        marker='o'
    )
    
    # Plot data setelah treatment
    after_df[col_name].plot(
        ax=ax,
        label='Setelah Treatment',
        color='green',
        alpha=0.7,
        marker='s'
    )
    
    ax.set_title(f'Perbandingan {col_name.upper()} Sebelum dan Setelah Treatment')
    ax.set_ylabel('Nilai')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Fungsi untuk plot forecast komparatif
def plot_forecast_comparison(before_df, after_df, before_pred, after_pred, col_name, forecast_days):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot data historis sebelum treatment
    before_df[col_name].plot(
        ax=ax,
        label='Sebelum Treatment (Historis)',
        color='blue',
        alpha=0.5,
        marker='o'
    )
    
    # Plot data historis setelah treatment
    after_df[col_name].plot(
        ax=ax,
        label='Setelah Treatment (Historis)',
        color='green',
        alpha=0.5,
        marker='s'
    )
    
    # Buat tanggal prediksi
    last_before_date = before_df.index[-1]
    future_before_dates = [last_before_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    last_after_date = after_df.index[-1]
    future_after_dates = [last_after_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Plot prediksi sebelum treatment
    ax.plot(
        future_before_dates,
        before_pred,
        label='Sebelum Treatment (Prediksi)',
        color='orange',
        linestyle='--',
        marker='o'
    )
    
    # Plot prediksi setelah treatment
    ax.plot(
        future_after_dates,
        after_pred,
        label='Setelah Treatment (Prediksi)',
        color='red',
        linestyle='--',
        marker='s'
    )
    
    # Garis pemisah
    ax.axvline(x=last_before_date, color='blue', linestyle=':', alpha=0.7)
    ax.axvline(x=last_after_date, color='green', linestyle=':', alpha=0.7)
    
    ax.set_title(f'Perbandingan Prediksi {col_name.upper()} {forecast_days} Hari')
    ax.set_ylabel('Nilai')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Fungsi utama Streamlit
def main():
    st.title('Water Quality Forecasting ')
    st.write("""
    Aplikasi prediksi kualitas air harian dengan perbandingan sebelum dan setelah treatment.
    Menampilkan prediksi untuk semua parameter (flow1, turbidity, pH, TDS).
    """)
    
    # Sidebar untuk upload data dan parameter
    with st.sidebar:
        st.header('Pengaturan')
        st.subheader("Upload Dataset")
        before_file = st.file_uploader("Upload dataset CSV (Sebelum Treatment)", type=['csv'])
        after_file = st.file_uploader("Upload dataset CSV (Setelah Treatment)", type=['csv'])
        
        st.subheader("Gemini API Configuration")
        api_key = st.text_input("Masukkan Gemini API Key", type="password")
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        forecast_days = st.slider(
            'Jumlah hari prediksi',
            min_value=1, max_value=30, value=7
        )
    
    if before_file is not None and after_file is not None:
        # Load dan bersihkan data
        before_df = pd.read_csv(before_file)
        after_df = pd.read_csv(after_file)
        
        before_clean = clean_data(before_df)
        after_clean = clean_data(after_df)
        
        # Tampilkan data
        st.subheader('Data Kualitas Air Harian')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sebelum Treatment**")
            st.dataframe(before_clean.tail(), use_container_width=True)
        with col2:
            st.markdown("**Setelah Treatment**")
            st.dataframe(after_clean.tail(), use_container_width=True)
        
        # Visualisasi komparatif untuk semua parameter
        st.subheader('Perbandingan Data Sebelum dan Setelah Treatment')
        
        for col in ['flow1', 'turbidity', 'ph', 'tds']:
            # Plot komparatif
            fig = plot_comparison(before_clean, after_clean, col)
            st.pyplot(fig)
            
            # Dapatkan insight komparatif
            insight_prompt = create_comparison_insight_prompt(before_clean, after_clean, col)
            insight_output = get_gemini_response(insight_prompt)
            
            if insight_output:
                with st.expander(f"Insight Perbandingan {col.upper()}"):
                    st.markdown(insight_output)
            
            st.markdown("---")
        
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = {
                'before': {},
                'after': {}
            }
        
        # Forecasting untuk semua parameter
        if st.button('Buat Prediksi untuk Semua Parameter'):
            if 'gemini_api_key' not in st.session_state:
                st.error("Harap masukkan Gemini API Key terlebih dahulu")
                st.stop()
                return
                
            with st.spinner('Membuat prediksi dan analisis...'):
                try:
                    # Buat container untuk hasil
                    results_container = st.container()
                    
                    # Buat prediksi untuk setiap parameter
                    for col in ['flow1', 'turbidity', 'ph', 'tds']:
                        with results_container:
                            st.subheader(f'Prediksi Parameter {col.upper()}')
                            
                            # Buat prediksi sebelum treatment
                            before_prompt = create_forecast_prompt(before_clean, col, forecast_days)
                            before_output = get_gemini_response(before_prompt)
                            before_pred = parse_prediction_output(before_output, forecast_days)
                            
                            # Buat prediksi setelah treatment
                            after_prompt = create_forecast_prompt(after_clean, col, forecast_days)
                            after_output = get_gemini_response(after_prompt)
                            after_pred = parse_prediction_output(after_output, forecast_days)
                            
                            if not before_pred or not after_pred:
                                st.error(f"Gagal memproses prediksi untuk {col}")
                                continue
                            
                            # Tampilkan visualisasi komparatif prediksi
                            fig = plot_forecast_comparison(
                                before_clean, after_clean, 
                                before_pred, after_pred,
                                col, forecast_days
                            )
                            st.pyplot(fig)
                            
                            # Tampilkan tabel prediksi
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Prediksi Sebelum Treatment**")
                                last_before_date = before_clean.index[-1]
                                future_before_dates = [last_before_date + timedelta(days=i+1) for i in range(forecast_days)]
                                before_forecast_df = pd.DataFrame({
                                    'Tanggal': future_before_dates,
                                    'Prediksi': before_pred
                                }).set_index('Tanggal')
                                st.dataframe(before_forecast_df.style.format({'Prediksi': '{:.4f}'}), 
                                            use_container_width=True)
                            
                            with col2:
                                st.markdown("**Prediksi Setelah Treatment**")
                                last_after_date = after_clean.index[-1]
                                future_after_dates = [last_after_date + timedelta(days=i+1) for i in range(forecast_days)]
                                after_forecast_df = pd.DataFrame({
                                    'Tanggal': future_after_dates,
                                    'Prediksi': after_pred
                                }).set_index('Tanggal')
                                st.dataframe(after_forecast_df.style.format({'Prediksi': '{:.4f}'}), 
                                            use_container_width=True)
                            
                            # Dapatkan insight sebelum treatment
                            before_insight_prompt = create_insight_prompt(before_clean, col, before_pred, "sebelum")
                            before_insight_output = get_gemini_response(before_insight_prompt)
                            
                            # Dapatkan insight setelah treatment
                            after_insight_prompt = create_insight_prompt(after_clean, col, after_pred, "setelah")
                            after_insight_output = get_gemini_response(after_insight_prompt)
                            
                            # Tampilkan insight
                            col1, col2 = st.columns(2)
                            with col1:
                                if before_insight_output:
                                    with st.expander(f"Insight Sebelum Treatment ({col.upper()})"):
                                        st.markdown(before_insight_output)
                            
                            with col2:
                                if after_insight_output:
                                    with st.expander(f"Insight Setelah Treatment ({col.upper()})"):
                                        st.markdown(after_insight_output)
                            
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")
    else:
        st.info("Silakan upload dataset CSV sebelum dan setelah treatment untuk memulai analisis.")

if __name__ == '__main__':
    main()