import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import base64

# Konfigurasi halaman
st.set_page_config(page_title="📊 Dashboard Kelulusan Mahasiswa", layout="wide", initial_sidebar_state="expanded")
if "welcome_sharingan" not in st.session_state:
    components.html(f"""
        <style>
        .zoom-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            animation: zoomIn 3s ease-out forwards;
        }}
        </style>

        <div class='zoom-container'>
            <img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnV6amMwM3U1cmc2YjZlbmJuMm8weWNoaXlqbWRjcGtqMWttYjY1ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/K9GWAtJoaaIy7vSCW8/giphy.gif' style='width: 80vw; max-width: 560px;'/>
        </div>
    """, height=700)

    st.session_state.welcome_sharingan = True
    st.rerun()
    st.stop()


# Sidebar Menu
menu = st.sidebar.radio("📂 Menu Navigasi", ["🏠 Beranda", "📁 Upload Dataset", "🧪 Prediksi", "📊 Evaluasi Model", "ℹ️ Tentang"])

# Variabel Global
if "model_dt" not in st.session_state:
    st.session_state.model_dt = None
    st.session_state.model_svm = None
    st.session_state.df = None
    st.session_state.scaler = None
    st.session_state.X_test = None
    st.session_state.y_test = None

# ----------------------- BERANDA -----------------------
if menu == "🏠 Beranda":
    st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")
    st.markdown("""
    Selamat datang di aplikasi prediksi kelulusan mahasiswa berdasarkan nilai dan kehadiran!  
    Di sini kamu bisa:
    - 📥 Upload data mahasiswa
    - 🤖 Melatih model Decision Tree & SVM
    - 🔮 Memprediksi kelulusan berdasarkan input nilai
    - 📊 Melihat hasil evaluasi model

    **Yuk mulai dari menu di sebelah kiri!** 👈
    """)

# ----------------------- UPLOAD DATA -----------------------
elif menu == "📁 Upload Dataset":
    st.title("📁 Upload Dataset Mahasiswa")
    uploaded_file = st.file_uploader("Unggah file CSV yang berisi data mahasiswa", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required = {"Nama", "NIM", "Kehadiran", "UTS", "UAS", "Tugas"}
        if not required.issubset(df.columns):
            st.error("❌ Kolom tidak sesuai! Harus ada: Nama, NIM, Kehadiran, UTS, UAS, Tugas")
        else:
            st.success("✅ Dataset berhasil dimuat!")
            df["Lulus"] = np.where(
                (df["Kehadiran"] >= 75) & (df["UTS"] >= 70) & (df["UAS"] >= 75) & (df["Tugas"] >= 70), 1, 0
            )

            fitur = ["Kehadiran", "UTS", "UAS", "Tugas"]
            X = df[fitur]
            y = df["Lulus"]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Latih model
            model_dt = DecisionTreeClassifier(random_state=42)
            model_dt.fit(X_train, y_train)

            model_svm = SVC(probability=True, random_state=42)
            model_svm.fit(X_train, y_train)

            # Simpan ke session_state
            st.session_state.df = df
            st.session_state.scaler = scaler
            st.session_state.model_dt = model_dt
            st.session_state.model_svm = model_svm
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.dataframe(df.head())
            st.markdown("Statistik Deskriptif:")
            st.dataframe(df.describe())

# ----------------------- PREDIKSI -----------------------
elif menu == "🧪 Prediksi":
    st.title("🧪 Prediksi Kelulusan Mahasiswa Baru")

    if st.session_state.model_dt and st.session_state.model_svm:
        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)
            with col1:
                kehadiran = st.slider("📅 Kehadiran (%)", 0, 100, 75)
                uts = st.slider("📝 Nilai UTS", 0, 100, 70)
            with col2:
                uas = st.slider("📚 Nilai UAS", 0, 100, 75)
                tugas = st.slider("🧾 Nilai Tugas", 0, 100, 75)

            threshold = st.slider("🎯 Threshold Kelulusan (%)", 0, 100, 50)
            submit = st.form_submit_button("🔮 Prediksi")

        if submit:
            input_data = np.array([[kehadiran, uts, uas, tugas]])
            input_scaled = st.session_state.scaler.transform(input_data)

            prob_dt = st.session_state.model_dt.predict_proba(input_scaled)[0][1]
            prob_svm = st.session_state.model_svm.predict_proba(input_scaled)[0][1]

            hasil_dt = "Lulus 🥳" if prob_dt >= threshold / 100 else "Tidak Lulus 😢"
            hasil_svm = "Lulus 🥳" if prob_svm >= threshold / 100 else "Tidak Lulus 😢"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌳 Decision Tree", hasil_dt)
                st.caption(f"Probabilitas: {prob_dt:.2f}")
            with col2:
                st.metric("🤖 SVM", hasil_svm)
                st.caption(f"Probabilitas: {prob_svm:.2f}")
    else:
        st.warning("⚠️ Silakan upload dan latih model terlebih dahulu di menu 'Upload Dataset'.")

# ----------------------- EVALUASI -----------------------
elif menu == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Akurasi Model")

    if st.session_state.model_dt and st.session_state.model_svm:
        y_test = st.session_state.y_test
        y_pred_dt = st.session_state.model_dt.predict(st.session_state.X_test)
        y_pred_svm = st.session_state.model_svm.predict(st.session_state.X_test)

        acc_dt = accuracy_score(y_test, y_pred_dt)
        acc_svm = accuracy_score(y_test, y_pred_svm)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌳 Decision Tree")
            st.write(f"Akurasi: **{acc_dt:.2f}**")
            fig1, ax1 = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, cmap="YlGn", ax=ax1)
            ax1.set_title("Confusion Matrix - Decision Tree")
            st.pyplot(fig1)

        with col2:
            st.subheader("🤖 SVM")
            st.write(f"Akurasi: **{acc_svm:.2f}**")
            fig2, ax2 = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap="PuBu", ax=ax2)
            ax2.set_title("Confusion Matrix - SVM")
            st.pyplot(fig2)

        st.markdown("### 📌 Kesimpulan Akurasi")
        if acc_dt > acc_svm:
            st.success("🏆 Decision Tree lebih unggul dalam dataset ini.")
        elif acc_svm > acc_dt:
            st.success("🏆 SVM lebih unggul dan lebih stabil dalam klasifikasi.")
        else:
            st.info("⚖️ Keduanya punya performa setara!")
    else:
        st.warning("⚠️ Data belum dimuat atau model belum dilatih.")

# ----------------------- TENTANG -----------------------
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk **Praktikum Data Mining 2025**  
    Program Studi Teknik Informatika — Universitas Pelita Bangsa

    - 🧠 Algoritma: Decision Tree & Support Vector Machine
    - 📈 Fitur: Upload dataset, prediksi kelulusan, evaluasi model
    - 💻 Dibuat dengan Python + Streamlit

    Dibuat oleh Kelompok 6:  
    - 👨‍💻 Rian Fauza Dinata — 312210083
    - 👨‍💻 Mohammad Azmi Abdussyukur — 312210109
    - 👨‍💻 Michael Toga Junior Sinaga — 312310774

    Terima kasih sudah mencoba aplikasi ini! Semoga bermanfaat ✨
    """)

