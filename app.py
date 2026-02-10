import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report


st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide"
)


@st.cache_resource
def load_model():
    try:
        model = joblib.load('telco_churn_model.pkl')
        return model
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # Basic cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        return df
    except:
        return None

model = load_model()

st.title("Aplikasi Prediksi Customer Churn")
st.write("""
Aplikasi ini menggunakan **Machine Learning (Random Forest + SMOTE)** untuk memprediksi 
apakah pelanggan telekomunikasi berpotensi berhenti berlangganan (Churn) atau tidak.
""")

if model is None:
    st.error("File 'telco_churn_model.pkl' tidak ditemukan. Pastikan file model ada di folder yang sama.")
else:

    st.sidebar.header("Masukkan Data Pelanggan")
    
    # Input Angka (Numerik)
    st.sidebar.subheader("Informasi Tagihan")
    tenure = st.sidebar.slider("Lama Berlangganan (Bulan)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Tagihan Bulanan ($)", min_value=0.0, max_value=150.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Tagihan ($)", min_value=0.0, value=monthly_charges * tenure)

    # Input Pilihan (Kategorikal) - NILAI HARUS SESUAI MAPPING DI COLAB
    st.sidebar.subheader("Informasi Layanan")
    
    # 1. Contract
    contract_opt = {'Bulanan (Month-to-month)': 0, 'Satu Tahun (One year)': 1, 'Dua Tahun (Two year)': 2}
    contract = st.sidebar.selectbox("Jenis Kontrak", list(contract_opt.keys()))
    
    # 2. Internet Service
    internet_opt = {'DSL': 0, 'Fiber Optic': 1, 'Tidak Ada': 2}
    internet = st.sidebar.selectbox("Layanan Internet", list(internet_opt.keys()))
    
    # 3. Online Security
    security_opt = {'Tidak': 0, 'Ya': 1, 'Tidak ada internet': 2}
    security = st.sidebar.selectbox("Keamanan Online", list(security_opt.keys()))
    
    # 4. Tech Support
    tech_opt = {'Tidak': 0, 'Ya': 1, 'Tidak ada internet': 2}
    tech = st.sidebar.selectbox("Dukungan Teknis", list(tech_opt.keys()))

    # Siapkan data untuk prediksi
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract_opt[contract]],
        'InternetService': [internet_opt[internet]],
        'OnlineSecurity': [security_opt[security]],
        'TechSupport': [tech_opt[tech]]
    })

    # Tombol Eksekusi
    if st.sidebar.button("Analisis Risiko Churn"):
        # Lakukan prediksi
        prediksi = model.predict(input_data)
        proba = model.predict_proba(input_data)
        
        # Ambil probabilitas Churn (kelas 1)
        chance_churn = proba[0][1] * 100
        chance_stay = proba[0][0] * 100

        # Tampilkan Hasil di Main Page
        st.subheader("Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        # Kartu visualisasi sederhana
        with col1:
            st.info("Detail Data Pelanggan")
            st.dataframe(input_data)
        
        with col2:
            st.info("Status Prediksi")
            if prediksi[0] == 1:
                st.error("BERISIKO CHURN (Berhenti)")
                st.write(f"Probabilitas Churn: **{chance_churn:.2f}%**")
                
                # Tambahan Detail Interpretasi Risiko
                if chance_churn > 80:
                    st.write("Tingkat Risiko: **Sangat Tinggi**")
                elif chance_churn > 60:
                     st.write("Tingkat Risiko: **Tinggi**")
                else:
                     st.write("Tingkat Risiko: **Sedang**")

                st.progress(int(chance_churn))
                
                # Saran Lebih Spesifik
                st.write("**Rekomendasi Tindakan:**")
                if tenure < 12:
                    st.write("- Pelanggan baru rentan pindah. Tawarkan program onboarding atau diskon selamat datang.")
                if contract_opt[contract] == 0: # Monthly
                    st.write("- Dorong pelanggan beralih ke kontrak jangka panjang (1 atau 2 tahun) dengan insentif harga.")
                if monthly_charges > 80:
                    st.write("- Evaluasi paket harga, tawarkan paket bundle yang lebih hemat.")
                
            else:
                st.success("SETIA (Tidak Berhenti)")
                st.write(f"Probabilitas Setia: **{chance_stay:.2f}%**")
                
                # Tambahan Detail Interpretasi Loyalitas
                if chance_stay > 80:
                    st.write("Tingkat Loyalitas: **Sangat Tinggi**")
                else:
                    st.write("Tingkat Loyalitas: **Cukup Baik**")

                st.progress(int(chance_stay))
                
                st.write("**Rekomendasi Tindakan:**")
                st.write("- Pertahankan kualitas layanan (Service Level Agreement).")
                if tenure > 24:
                    st.write("- Tawarkan program loyalitas atau reward sebagai apresiasi.")


    
    # --- BAGIAN TAMBAHAN: INFO MODEL & PLOT ---
    st.markdown("---")
    st.subheader("Insight Model & Data")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Evaluasi Model", "Visualisasi Data"])

    with tab1:
        st.write("### Faktor Penentu Prediksi (Feature Importance)")
        st.write("Grafik berikut menunjukkan fitur mana yang paling berpengaruh dalam memprediksi Churn.")
        
        if hasattr(model, 'feature_importances_'):
            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
            feat_chart = feat_importances.sort_values(ascending=True) # Sort for better chart
            st.bar_chart(feat_chart)
        else:
            st.warning("Model ini tidak mendukung fitur importance.")

    with tab2:
        st.write("### Skor Akurasi Model")
        
        df = load_data()
        
        if df is not None:
            try:
                # Create copy for evaluation
                df_eval = df.copy()
                
                # Mapping Contract
                map_contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
                df_eval['Contract'] = df_eval['Contract'].map(map_contract)
                
                # Mapping InternetService
                map_internet = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
                df_eval['InternetService'] = df_eval['InternetService'].map(map_internet)
                
                # Mapping OnlineSecurity
                map_security = {'No': 0, 'Yes': 1, 'No internet service': 2}
                df_eval['OnlineSecurity'] = df_eval['OnlineSecurity'].map(map_security)
                
                # Mapping TechSupport
                map_tech = {'No': 0, 'Yes': 1, 'No internet service': 2}
                df_eval['TechSupport'] = df_eval['TechSupport'].map(map_tech)
                
                # Label Churn
                df_eval['Churn_Label'] = df_eval['Churn'].map({'Yes': 1, 'No': 0})
                
                # Filter features
                X_eval = df_eval[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']]
                y_eval_true = df_eval['Churn_Label']
                
                # Drop NA
                combined = pd.concat([X_eval, y_eval_true], axis=1).dropna()
                X_eval = combined.drop('Churn_Label', axis=1)
                y_eval_true = combined['Churn_Label']

                # Hitung Score
                y_pred_eval = model.predict(X_eval)
                acc = accuracy_score(y_eval_true, y_pred_eval)
                f1 = f1_score(y_eval_true, y_pred_eval)
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Akurasi", f"{acc*100:.2f}%")
                col_m2.metric("F1 Score", f"{f1*100:.2f}%")
                
                st.write("Classification Report:")
                st.code(classification_report(y_eval_true, y_pred_eval))
                
                st.write("Skor ini dihitung berdasarkan seluruh data dataset.")
                
                # Tampilkan Distribusi Churn
                st.write("#### Distribusi Churn Global")
                churn_counts = df['Churn'].value_counts()
                st.bar_chart(churn_counts)
                
            except Exception as e:
                st.error(f"Error evaluasi: {e}")
        else:
            st.error("Gagal memuat dataset. Pastikan file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' ada.")

    with tab3:
        st.write("### Visualisasi Hubungan Data")
        st.caption("Eksplorasi interaktif hubungan antar variabel dengan status Churn.")
        
        df = load_data()
        if df is not None:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("#### 1. Pengaruh Jenis Kontrak")
                st.caption("Distribusi pelanggan Churn berdasarkan tipe kontrak.")
                ct_contract = pd.crosstab(df['Contract'], df['Churn'])
                st.bar_chart(ct_contract)
                st.info("💡 **Insight:** Pelanggan dengan kontrak **Bulanan (Month-to-month)** memiliki tingkat Churn yang jauh lebih tinggi dibandingkan kontrak tahunan. Kontrak jangka panjang meningkatkan retensi.")
                
            with col_b:
                st.write("#### 2. Layanan Internet")
                st.caption("Perbandingan Churn antara pengguna DSL dan Fiber Optic.")
                ct_internet = pd.crosstab(df['InternetService'], df['Churn'])
                st.bar_chart(ct_internet)
                st.info("💡 **Insight:** Pengguna **Fiber Optic** cenderung lebih banyak berpindah (Churn) dibandingkan DSL, kemungkinan karena faktor harga atau persaingan layanan di segmen ini.")
            
            st.markdown("---")
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.write("#### 3. Metode Pembayaran")
                st.caption("Apakah metode pembayaran mempengaruhi loyalitas?")
                ct_payment = pd.crosstab(df['PaymentMethod'], df['Churn'])
                st.bar_chart(ct_payment)
                st.info("💡 **Insight:** Pelanggan yang menggunakan **Electronic Check** memiliki kecenderungan Churn tertinggi dibandingkan metode pembayaran otomatis lainnya.")

            with col_d:
                 st.write("#### 4. Dukungan Teknis (Tech Support)")
                 st.caption("Peran fitur tambahan terhadap retensi pelanggan.")
                 ct_tech = pd.crosstab(df['TechSupport'], df['Churn'])
                 st.bar_chart(ct_tech)
                 st.info("💡 **Insight:** Ketidakhadiran layanan **Tech Support** sangat berkorelasi dengan keputusan pelanggan untuk berhenti berlangganan.")

            st.markdown("---")
            st.write("#### 5. Loyalitas Berdasarkan Waktu (Tenure)")
            st.caption("Bagaimana lama berlangganan mempengaruhi risiko Churn.")
            
            df_viz = df.copy()
            # Binning tenure
            df_viz['Tenure Group'] = pd.cut(df_viz['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=['0-1 Thn', '1-2 Thn', '2-4 Thn', '4-5 Thn', '> 5 Thn'])
            ct_tenure = pd.crosstab(df_viz['Tenure Group'], df_viz['Churn'])
            st.bar_chart(ct_tenure)
            st.info("💡 **Insight:** Pelanggan baru (**< 1 Tahun**) berada pada fase paling kritis. Risiko Churn menurun drastis setelah pelanggan melewati tahun pertama layanan.")

        else:
            st.warning("Data tidak tersedia.")

# Footer
st.markdown("---")
st.caption("Capstone Project Machine Learning | Telco Dataset")