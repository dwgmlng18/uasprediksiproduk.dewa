import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coffee Shop - Prediksi Produk Terlaris", layout="wide")
st.title("Prediksi Kategori Produk yang Paling Sering Dibeli Pelanggan")
st.markdown("### Metode: K-Nearest Neighbors + **Euclidean Distance** (Cepat & Akurat!)")

@st.cache_data
def load_data():
    return pd.read_excel("Coffee Shop Sales.xlsx", sheet_name="Transactions")

df = load_data()

st.subheader("1. Data Transaksi Awal")
n_show = st.radio(
    "Pilih berapa baris data awal yang ingin ditampilkan:",
    options=["50 baris", "200 baris", "500 baris", "Semua data (~149.116 baris)"],
    index=1,
    horizontal=True
)

if n_show == "Semua data (~149.116 baris)":
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(int(n_show.split()[0])), use_container_width=True)

le_loc = LabelEncoder()
df["store_location_enc"] = le_loc.fit_transform(df["store_location"])

X_full = df[["transaction_qty", "unit_price", "store_location_enc", "product_id", "Month", "Weekday", "Hour"]]
y_full = df["product_category"]

le_cat = LabelEncoder()
y_enc_full = le_cat.fit_transform(y_full)

st.subheader("2. Pilih dan Tampilkan Jumlah Data untuk Training dan Testing")
train_size = st.radio(
    "Gunakan berapa data untuk training?",
    options=["1.000 baris", "5.000 baris", "20.000 baris", "50.000 baris", "Semua data (~149.116)"],
    index=4,
    horizontal=True
)

n_train = {
    "1.000 baris": 1000,
    "5.000 baris": 5000,
    "20.000 baris": 20000,
    "50.000 baris": 50000,
    "Semua data (~149.116)": len(X_full)
}[train_size]

X = X_full.head(n_train)
y_enc = y_enc_full[:n_train]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

view = st.radio(
    "Pilih data yang ingin Anda lihat:",
    options=["Data Training (Semua Baris)", "Data Testing (Semua Baris)"],
    horizontal=True
)

if view == "Data Training (Semua Baris)":
    train_view = X_train.copy()
    train_view["Kategori Aktual"] = le_cat.inverse_transform(y_train)
    st.write(f"**Total Data Training: {len(train_view):,} baris**")
    st.dataframe(train_view.reset_index(drop=True), use_container_width=True, height=600)
else:
    test_view = X_test.copy()
    test_view["Kategori Aktual"] = le_cat.inverse_transform(y_test)
    st.write(f"**Total Data Testing: {len(test_view):,} baris**")
    st.dataframe(test_view.reset_index(drop=True), use_container_width=True, height=600)

st.subheader("3. Pilih Nilai K yang Ingin Ditampilkan")
k_selected = st.radio(
    "Pilih nilai K (akan dibandingkan dengan K terbaik):",
    options=[3, 5, 7, 9, 11, 13, 15],
    index=3,
    horizontal=True
)

st.info(f"Kamu memilih **K = {k_selected}** → akan dibandingkan dengan K terbaik dari eksperimen")

run_knn = st.button("PROSES KNN & EVALUASI KLASIFIKASI LENGKAP", type="primary", use_container_width=True)

if run_knn:
    with st.spinner("Menguji semua nilai K dari 1 hingga 15..."):
        k_list = [1, 3, 5, 7, 9, 11, 13, 15]
        results = []
        predictions_dict = {}
        best_k = 1
        best_acc = 0

        for k in k_list:
            model = KNeighborsClassifier(
                n_neighbors=k,
                metric='euclidean',
                n_jobs=-1,
                algorithm='auto'
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            predictions_dict[k] = pred
            results.append({
                "K": k,
                "Akurasi": f"{acc*100:.3f}%",
                "Prediksi Benar": f"{int(acc * len(y_test)):,}",
                "Total Testing": len(y_test)
            })
            if acc > best_acc:
                best_acc = acc
                best_k = k

        results_df = pd.DataFrame(results)

        def highlight_rows(row):
            if row["K"] == k_selected and row["K"] == best_k:
                return ['background-color: #ffff99; font-weight: bold'] * len(row)
            elif row["K"] == k_selected:
                return ['background-color: #ffff99; font-weight: bold'] * len(row)
            elif row["K"] == best_k:
                return ['background-color: #90EE90'] * len(row)
            else:
                return [''] * len(row)

        st.success("PROSES KNN SELESAI!")
        st.subheader("Perbandingan Akurasi Semua Nilai K")
        st.dataframe(results_df.style.apply(highlight_rows, axis=1), use_container_width=True)

        pred_user = predictions_dict[k_selected]
        pred_best = predictions_dict[best_k]
        true_labels = le_cat.inverse_transform(y_test)
        label_user = le_cat.inverse_transform(pred_user)
        label_best = le_cat.inverse_transform(pred_best)

        st.markdown("---")
        st.header("Hasil Prediksi Lengkap")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"K = {k_selected} (Pilihan Anda)")
            tbl1 = pd.DataFrame({
                "No": range(1, len(label_user)+1),
                "Prediksi": label_user,
                "Aktual": true_labels,
                "Status": ["Benar" if p==a else "Salah" for p,a in zip(label_user, true_labels)]
            })
            st.metric("Akurasi", f"{accuracy_score(y_test, pred_user)*100:.3f}%")
            st.dataframe(tbl1, use_container_width=True, height=600)

        with col2:
            st.subheader(f"K = {best_k} (MODEL TERBAIK)")
            tbl2 = pd.DataFrame({
                "No": range(1, len(label_best)+1),
                "Prediksi": label_best,
                "Aktual": true_labels,
                "Status": ["Benar" if p==a else "Salah" for p,a in zip(label_best, true_labels)]
            })
            st.metric("Akurasi Tertinggi", f"{best_acc*100:.3f}%",
                      delta=f"{(best_acc - accuracy_score(y_test, pred_user))*100:+.3f}%")
            st.dataframe(tbl2, use_container_width=True, height=600)

        st.markdown("---")
        st.header("Evaluasi Klasifikasi Lengkap (Model Terbaik K = {})".format(best_k))

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Classification Report")
            report = classification_report(true_labels, label_best, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
            st.write("**Interpretation:**")
            st.write("- **Precision**: Seberapa banyak dari yang diprediksi positif yang benar-benar positif")
            st.write("- **Recall**: Seberapa banyak kasus positif yang berhasil ditemukan")
            st.write("- **F1-Score**: Rata-rata harmonik dari precision & recall")

        with colB:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(true_labels, label_best)
            fig, ax = plt.subplots(figsize=(11, 9))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=le_cat.classes_, yticklabels=le_cat.classes_)
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            plt.title(f"Confusion Matrix - Model Terbaik (K={best_k})")
            st.pyplot(fig)

        st.subheader("Analisis Pola Kesalahan")
        errors = pd.DataFrame({"Aktual": true_labels, "Prediksi": label_best})
        wrong = errors[errors["Aktual"] != errors["Prediksi"]]
        top_mistakes = wrong.groupby(["Aktual", "Prediksi"]).size().sort_values(ascending=False).head(10)

        if len(wrong) > 0:
            st.write(f"**Total kesalahan: {len(wrong):,} dari {len(y_test):,} ({len(wrong)/len(y_test)*100:.2f}%)**")
            st.write("**10 Kesalahan Prediksi Terbanyak:**")
            st.dataframe(top_mistakes.reset_index(name="Jumlah Kesalahan"))
            st.info(f"""
            **Rekomendasi Perbaikan:**
            - Kelas **{wrong['Aktual'].mode().iloc[0]}** paling sering salah → pertimbangkan menambah fitur seperti ukuran cup, suhu minuman, atau waktu promo
            - Jika 'Coffee' sering diprediksi jadi 'Tea', mungkin fitur harga & qty saja belum cukup kuat
            - Coba teknik oversampling/undersampling jika ada kelas tidak seimbang
            """)
        else:
            st.success("SEMUA PREDIKSI BENAR!")

        st.balloons()

else:
    st.info("Pilih K di atas → klik tombol biru untuk melihat hasil lengkap + evaluasi klasifikasi")

st.caption("Coffee Shop Product Prediction | KNN + Evaluasi Klasifikasi Lengkap (Precision, Recall, F1, Confusion Matrix) | 2025")
