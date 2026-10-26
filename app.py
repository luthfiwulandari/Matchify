import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as rt

# --- 1. KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="Matchify - Rekomendasi Paket Telco AI",
    page_icon="📱",
    layout="wide"
)

# --- STYLING CSS KUSTOM (LATAR BELAKANG PUTIH BERSIH) ---
st.markdown("""
    <style>
    /* Mengatur background utama halaman menjadi putih bersih */
    .stApp {
        background-color: #FFFFFF;
    }
    .block-container {
        padding-top: 1rem;
        background-color: #FFFFFF;
    }
    /* Header Utama dengan warna #606376 */
    .matchify-header {
        background-color: #606376;
        padding: 22px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        letter-spacing: 1px;
        border-radius: 0px 0px 10px 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    /* Sub-Judul Section */
    .section-title {
        text-align: center;
        color: #606376;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }
    /* Kotak Kartu Hasil Rekomendasi di atas background putih */
    .matchify-card {
        background-color: #F8FAFC;
        padding: 22px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border-left: 7px solid #606376;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .badge-rank {
        background-color: #E2E8F0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-score {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        float: right;
    }
    </style>
""", unsafe_allow_html=True)

# --- RENDER HEADER & SUB-JUDUL ---
st.markdown('<div class="matchify-header">Matchify</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">GENERATE OFFER RECOMMENDATION</div>', unsafe_allow_html=True)

# --- 2. LOAD MODEL ONNX ---
@st.cache_resource
def load_onnx_model():
    session = rt.InferenceSession("rf_telco_recommendation.onnx")
    return session

try:
    onnx_session = load_onnx_model()
except Exception as e:
    st.error(f"Gagal memuat model ONNX. Pastikan file 'rf_telco_recommendation.onnx' ada di folder yang sama! Error: {e}")

# --- 3. KAMUS DETAIL REKOMENDASI PAKET (HARGA RESMI PASTI) ---
package_details = {
    "General Offer": {
        "title": "🎁 General Offer (Paket Reguler Harian/Mingguan)",
        "price": "Rp 50.000",
        "quota": "10 GB (All-Network)",
        "time": "Berlaku 24 Jam (30 Hari)",
        "benefit": "Cocok untuk penggunaan seluler standar sehari-hari (chatting & browsing ringan).",
        "reasoning": "Pola penggunaan data dan pengeluaran bulanan Anda berada pada kategori reguler."
    },
    "Device Upgrade Offer": {
        "title": "📱 Device Upgrade Offer (Bundling Smartphone)",
        "price": "Rp 150.000 / bulan",
        "quota": "30 GB + Unlimited Sosmed",
        "time": "Berlaku 24 Jam + Kontrak 12 Bulan",
        "benefit": "Diskon cicilan smartphone baru + kuota melimpah setiap bulan.",
        "reasoning": "Merek perangkat dan tingkat pengeluaran bulanan Anda memenuhi kriteria program loyalitas perangkat."
    },
    "Data Booster": {
        "title": "⚡ Data Booster Pack (Tambahan Kuota Utama)",
        "price": "Rp 25.000",
        "quota": "15 GB Extra Kuota Cepat",
        "time": "Berlaku 7 Hari",
        "benefit": "Kecepatan maksimum tanpa pembagian jaringan.",
        "reasoning": "Rata-rata penggunaan data bulanan Anda tergolong tinggi dan membutuhkan ekstra kuota."
    },
    "Retention Offer": {
        "title": "🤝 Special Retention Offer (Loyalty Bonus)",
        "price": "Rp 70.000",
        "quota": "25 GB + Ekstra Pulsa Telpon",
        "time": "Berlaku 30 Hari",
        "benefit": "Penawaran khusus untuk mencegah perpindahan operator dengan benefit berlipat.",
        "reasoning": "Berdasarkan riwayat keluhan dan masa aktif, Anda layak mendapatkan apresiasi khusus pelanggan setia."
    },
    "Top-up Promo": {
        "title": "💳 Top-up Cashback Promo",
        "price": "Rp 50.000 (Minimum Isi Ulang)",
        "quota": "Bonus Kuota 20% setiap isi ulang",
        "time": "Berlaku 7 Hari",
        "benefit": "Cashback saldo digital dan bonus kuota aplikasi.",
        "reasoning": "Frekuensi isi ulang Anda cukup sering dan responsif terhadap promo isi ulang."
    },
    "Streaming Partner Pack": {
        "title": "🎬 Streaming Partner Pack (Entertainment)",
        "price": "Rp 60.000",
        "quota": "10 GB Utama + 25 GB Kuota Khusus Aplikasi Video",
        "time": "Berlaku 30 Hari",
        "benefit": "Gratis langganan platform streaming video populer.",
        "reasoning": "Persentase penggunaan video Anda sangat tinggi, sangat cocok dengan paket hiburan ini."
    },
    "Roaming Pass": {
        "title": "✈️ International Roaming Pass",
        "price": "Rp 250.000",
        "quota": "10 GB Roaming Luar Negeri",
        "time": "Berlaku 7 Hari",
        "benefit": "Koneksi stabil tanpa ganti kartu saat bepergian ke luar negeri.",
        "reasoning": "Skor perjalanan Anda menunjukkan mobilitas tinggi antar wilayah."
    },
    "Family Plan Offer": {
        "title": "👨‍👩‍👧 Family Sharing Plan",
        "price": "Rp 200.000 / bulan",
        "quota": "80 GB (Dapat dibagi hingga 4 anggota keluarga)",
        "time": "Berlaku 30 Hari",
        "benefit": "Satu tagihan untuk seluruh anggota keluarga dengan kontrol kuota terpusat.",
        "reasoning": "Karakteristik penggunaan bulanan Anda sangat ideal untuk pengelolaan paket kolektif keluarga."
    },
    "Voice Bundle": {
        "title": "📞 Voice & Call Unlimited Bundle",
        "price": "Rp 35.000",
        "quota": "Unlimited Telepon ke Semua Operator + 2 GB Kuota",
        "time": "Berlaku 30 Hari",
        "benefit": "Bebas biaya panggilan suara lama tanpa takut pulsa habis.",
        "reasoning": "Durasi panggilan rata-rata Anda lebih dominan dibandingkan penggunaan data seluler."
    }
}

labels_mapping = {
    0: "Data Booster",
    1: "Device Upgrade Offer",
    2: "Family Plan Offer",
    3: "General Offer",
    4: "Retention Offer",
    5: "Roaming Pass",
    6: "Streaming Partner Pack",
    7: "Top-up Promo",
    8: "Voice Bundle"
}

# --- 4. FORM INPUT UTAMA (2 KOLOM SEJAJAR, KOSONG DENGAN PLACEHOLDER) ---
with st.form("recommendation_form"):
    col1, col2 = st.columns(2)

    with col1:
        plan_type = st.selectbox("Payment Type (Prabayar/Pascabayar):", ["Prepaid", "Postpaid"])
        avg_data_usage_gb = st.number_input("Average Data Usage (GB):", min_value=0.0, value=None, step=0.5, placeholder="Contoh: 5.0")
        avg_call_duration = st.number_input("Average call duration (minutes):", min_value=0.0, value=None, step=1.0, placeholder="Contoh: 10.0")
        monthly_spend = st.number_input("Monthly Expenses (Rp):", min_value=0, value=None, step=10000, placeholder="Contoh: 100000")
        travel_score = st.number_input("Travel Score (0.0 - 1.0):", min_value=0.0, max_value=1.0, value=None, step=0.01, placeholder="Contoh: 0.28")

    with col2:
        brand_device = st.selectbox("Device brand:", ["Samsung", "Xiaomi", "Apple", "Oppo", "Vivo", "Realme", "Huawei"])
        pct_video_usage = st.number_input("Video usage percentage (0-100):", min_value=0.0, max_value=100.0, value=None, step=1.0, placeholder="Contoh: 40")
        sms_freq = st.number_input("SMS Frequency:", min_value=0, value=None, step=1, placeholder="Contoh: 15")
        topup_freq = st.number_input("Topup Frequency:", min_value=0, value=None, step=1, placeholder="Contoh: 3")
        complaint_count = st.number_input("Jumlah Keluhan:", min_value=0, value=None, step=1, placeholder="Contoh: 0")

    st.markdown("")
    submitted = st.form_submit_button("🚀 Generate Top-5 Recommendations", use_container_width=True)

# --- 5. PROSES PREDIKSI & PENANGANAN ANOMALI (CLIPPING & SANITY CHECK) ---
if submitted:
    # Validasi jika ada input yang masih kosong
    if None in [avg_data_usage_gb, avg_call_duration, monthly_spend, travel_score, pct_video_usage, sms_freq, topup_freq, complaint_count]:
        st.warning("⚠️ Mohon isi semua kolom data terlebih dahulu sebelum melakukan generate rekomendasi!")
        st.stop()

    # Deteksi anomali untuk peringatan visual
    anomalies = []
    if monthly_spend < 10000:
        anomalies.append("💰 Pengeluaran bulanan (Monthly Expenses) terlalu kecil dari batas wajar normal.")
    if avg_data_usage_gb == 0 and pct_video_usage > 0:
        anomalies.append("📊 Penggunaan data 0 GB tetapi persentase video tinggi (tidak konsisten).")

    if anomalies:
        with st.container():
            st.warning("⚠️ **Catatan Sistem (Deteksi Nilai Ekstrem/Anomali):**\n" + "\n".join([f"- {a}" for a in anomalies]) + "\n\n_Sistem telah melakukan penyesuaian otomatis (clipping) pada batas wajar agar hasil prediksi model Random Forest tetap optimal._")

    # Terapkan Clipping / Batas Wajar Otomatis
    safe_monthly_spend = max(monthly_spend, 10000.0)
    safe_data_usage = max(avg_data_usage_gb, 0.5)
    normalized_video_usage = pct_video_usage / 100.0 if pct_video_usage > 1.0 else pct_video_usage

    # Struktur 11 fitur sesuai training model ONNX
    encoded_input = pd.DataFrame(0.0, index=[0], columns=[
        'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration', 'sms_freq',
        'monthly_spend', 'topup_freq', 'travel_score', 'complaint_count', 'avg_similarity',
        'plan_type_Prepaid', 'brand_device_Xiaomi'
    ])

    encoded_input['avg_data_usage_gb'] = safe_data_usage
    encoded_input['pct_video_usage'] = normalized_video_usage
    encoded_input['avg_call_duration'] = avg_call_duration
    encoded_input['sms_freq'] = sms_freq
    encoded_input['monthly_spend'] = safe_monthly_spend
    encoded_input['topup_freq'] = topup_freq
    encoded_input['travel_score'] = travel_score
    encoded_input['complaint_count'] = complaint_count
    encoded_input['avg_similarity'] = 0.5

    if plan_type == "Prepaid":
        encoded_input['plan_type_Prepaid'] = 1.0
    if brand_device == "Xiaomi":
        encoded_input['brand_device_Xiaomi'] = 1.0

    X_infer = encoded_input.to_numpy(dtype=np.float32)

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: X_infer})
    
    probabilities = outputs[1][0] 
    class_probs = [(i, prob) for i, prob in enumerate(probabilities)]
    sorted_classes = sorted(class_probs, key=lambda x: x[1], reverse=True)
    top_5_recommendations = sorted_classes[:5]

    st.success("Analisis Berhasil Dilakukan!")
    st.markdown("### 🏆 Top-5 Rekomendasi Paket untuk Anda")
    
    # Menampilkan Top-5 langsung terbuka dengan kartu yang selaras
    for rank, (class_idx, prob) in enumerate(top_5_recommendations, 1):
        pkg_name = labels_mapping.get(int(class_idx), "General Offer")
        details = package_details.get(pkg_name, package_details["General Offer"])
        
        st.markdown(f"""
            <div class="matchify-card">
                <div>
                    <span class="badge-rank">Peringkat #{rank}</span>
                    <span class="badge-score">Akurasi: {prob*100:.1f}%</span>
                </div>
                <h3 style="margin: 12px 0px 8px 0px; color: #1F2937; font-size: 18px;">{details['title']}</h3>
                <hr style="margin: 8px 0px; border: none; border-top: 1px solid #E2E8F0;">
                <p style="margin: 4px 0px; font-size: 15px; color: #606376; font-weight: bold;">💰 Harga Resmi: {details['price']}</p>
                <p style="margin: 4px 0px; font-size: 14px; color: #4B5563;">📦 <b>Kuota / Benefit:</b> {details['quota']}</p>
                <p style="margin: 4px 0px; font-size: 14px; color: #4B5563;">⏰ <b>Masa Aktif:</b> {details['time']}</p>
                <p style="margin: 4px 0px; font-size: 14px; color: #4B5563;">✨ <b>Keunggulan:</b> {details['benefit']}</p>
                <p style="margin: 8px 0px 0px 0px; font-size: 13px; color: #6B7280; font-style: italic; background-color: #FFFFFF; padding: 8px; border-radius: 6px; border: 1px solid #E2E8F0;">💡 <b>Alasan Sistem:</b> {details['reasoning']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #606376; font-size: 14px;'>Matchify © 2026 • All Rights Reserved</p>", unsafe_allow_html=True)
