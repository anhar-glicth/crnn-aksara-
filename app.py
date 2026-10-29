import streamlit as st
import os
import json
import torch
import cv2
import numpy as np
import io
import base64
from PIL import Image

import importlib
import renderer as renderer_module
importlib.reload(renderer_module)
from renderer import BimaAksaraRenderer

from model import CRNN
from translator import translate_indonesia_to_bima, BIMA_LEXICON
from syllabifier import bima_to_syllables

st.set_page_config(
    page_title="Penerbit Aksara — Preservasi Aksara Bima",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-page: #fff3ed;
  --bg-card: #f9ede6;
  --bg-white: #ffffff;
  --ink-black: #171412;
  --ink-dark: #2c2522;
  --ink-muted: #736b65;
  --ink-light: #9e958e;
  
  --accent-rust: #b3391b;
  --accent-gold: #c68a4c;
  --border-editorial: rgba(44, 37, 34, 0.12);
  --border-fine: rgba(44, 37, 34, 0.08);

  --font-serif: 'Newsreader', Garamond, 'Times New Roman', serif;
  --font-sans: 'Plus Jakarta Sans', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: var(--bg-page) !important;
  color: var(--ink-dark) !important;
  font-family: var(--font-sans) !important;
  line-height: 1.6;
}

[data-testid="stSidebar"] {
  background-color: var(--bg-card) !important;
  border-right: 1px solid var(--border-editorial) !important;
}

.editorial-topbar {
  border-bottom: 1px solid var(--border-editorial);
  padding: 1.25rem 0;
  margin-bottom: 2.2rem;
  background-color: rgba(255, 243, 237, 0.96);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}

.brand-title {
  font-family: var(--font-serif);
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--ink-black);
  letter-spacing: -0.02em;
  text-decoration: none;
}

.brand-badge {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--accent-rust);
  font-weight: 600;
  margin-left: 0.8rem;
}

.header-colophon {
  font-size: 0.84rem;
  color: var(--ink-muted);
  font-style: italic;
  font-family: var(--font-serif);
}

.hero-editorial {
  padding: 0.5rem 0 2.5rem;
  text-align: center;
  max-width: 860px;
  margin: 0 auto;
}

.hero-manifesto-pre {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.22em;
  color: var(--accent-rust);
  margin-bottom: 0.85rem;
  font-weight: 600;
}

.hero-manifesto-title {
  font-family: var(--font-serif);
  font-size: clamp(2.4rem, 4.5vw, 3.8rem);
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--ink-black);
  line-height: 1.18;
  margin-bottom: 1.25rem;
}

.hero-manifesto-title em {
  font-style: italic;
  color: var(--accent-rust);
}

.hero-manifesto-lead {
  font-size: 1.05rem;
  color: var(--ink-muted);
  line-height: 1.7;
  max-width: 680px;
  margin: 0 auto 2rem;
}

.colophon-specs-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  padding: 1.5rem 0;
  border-top: 1px solid var(--border-editorial);
  border-bottom: 1px solid var(--border-editorial);
  margin-bottom: 2.5rem;
  text-align: left;
}

.colophon-item h5 {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--ink-light);
  margin-bottom: 0.25rem;
  font-family: var(--font-sans);
}

.colophon-item p {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--ink-black);
  font-family: var(--font-serif);
  margin: 0;
}

.editorial-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-editorial);
  border-radius: 4px;
  padding: 1.8rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 6px rgba(44, 37, 34, 0.03);
}

.editorial-card-white {
  background-color: var(--bg-white);
  border: 1px solid var(--border-editorial);
  border-radius: 4px;
  padding: 1.8rem;
  margin-bottom: 1.5rem;
}

.editorial-section-title {
  font-family: var(--font-serif);
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--ink-black);
  letter-spacing: -0.01em;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
}

.section-number {
  font-size: 0.85rem;
  font-family: var(--font-sans);
  color: var(--accent-rust);
  font-weight: 700;
}

.editorial-caption {
  font-size: 0.88rem;
  color: var(--ink-muted);
  margin-bottom: 1.25rem;
  line-height: 1.5;
}

.syllable-token {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background-color: var(--bg-white);
  border: 1px solid var(--border-editorial);
  color: var(--ink-black);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  font-size: 0.85rem;
  padding: 4px 10px;
  border-radius: 3px;
  margin-right: 6px;
  margin-bottom: 6px;
}

.word-lexicon-tag {
  display: inline-block;
  background-color: #f1f8f4;
  border: 1px solid #c2e2cf;
  color: #1e5e3a;
  font-size: 0.84rem;
  padding: 3px 8px;
  border-radius: 3px;
  margin-right: 6px;
  margin-bottom: 4px;
}

.word-oov-tag {
  display: inline-block;
  background-color: #fdf5f2;
  border: 1px solid #ebd3c8;
  color: var(--accent-rust);
  font-size: 0.84rem;
  padding: 3px 8px;
  border-radius: 3px;
  margin-right: 6px;
  margin-bottom: 4px;
}

.manuscript-mat {
  background-color: #fbf6f0;
  border: 1px solid var(--border-editorial);
  border-radius: 4px;
  padding: 2.2rem 1.5rem;
  text-align: center;
  box-shadow: inset 0 0 15px rgba(44, 37, 34, 0.05);
  margin: 1.2rem 0;
}

div.stButton > button {
  background-color: #ffffff !important;
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
  border: 1.5px solid rgba(44, 37, 34, 0.22) !important;
  border-radius: 4px !important;
  font-family: var(--font-serif) !important;
  font-style: italic !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  padding: 0.6rem 1.25rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 4px rgba(44, 37, 34, 0.04) !important;
}

div.stButton > button:hover {
  background-color: var(--accent-rust) !important;
  border-color: var(--accent-rust) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  transform: translateY(-1px) !important;
}

div.stButton > button:active {
  transform: translateY(0) !important;
}

div[data-testid="stDownloadButton"] > button {
  background-color: var(--accent-rust) !important;
  border: 1px solid var(--accent-rust) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-family: var(--font-sans) !important;
  font-style: normal !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 0.7rem 1.6rem !important;
  border-radius: 4px !important;
  box-shadow: 0 2px 6px rgba(179, 57, 27, 0.25) !important;
}

div[data-testid="stDownloadButton"] > button:hover {
  background-color: #8c2810 !important;
  border-color: #8c2810 !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
label,
label *,
.stTextInput label,
.stTextInput label *,
.stSelectbox label,
.stSelectbox label *,
.stSlider label,
.stSlider label *,
.stFileUploader label,
.stFileUploader label * {
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
  font-family: var(--font-sans) !important;
  font-size: 0.84rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  margin-bottom: 0.35rem !important;
  opacity: 1 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 2.2rem !important;
  background-color: transparent !important;
  border-bottom: 1px solid var(--border-editorial) !important;
  padding: 0 !important;
  margin-bottom: 1.8rem !important;
}

.stTabs [data-baseweb="tab"] {
  height: 48px !important;
  background-color: transparent !important;
  border: none !important;
  padding: 0 4px !important;
  opacity: 1 !important;
}

.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div,
[data-testid="stTab"],
[data-testid="stTab"] * {
  font-family: var(--font-serif) !important;
  font-size: 1.15rem !important;
  font-weight: 600 !important;
  color: #2c2522 !important;
  -webkit-text-fill-color: #2c2522 !important;
  opacity: 1 !important;
  transition: color 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover * {
  color: var(--accent-rust) !important;
  -webkit-text-fill-color: var(--accent-rust) !important;
}

.stTabs [aria-selected="true"] {
  border-bottom: 2px solid var(--accent-rust) !important;
}

.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] * {
  color: var(--accent-rust) !important;
  -webkit-text-fill-color: var(--accent-rust) !important;
  font-weight: 700 !important;
}

.stTextInput input,
.stTextArea textarea {
  background-color: #ffffff !important;
  border: 1px solid rgba(44, 37, 34, 0.25) !important;
  border-radius: 4px !important;
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
  font-family: var(--font-sans) !important;
  font-size: 0.98rem !important;
  font-weight: 500 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
  border-color: #171412 !important;
  box-shadow: 0 0 0 1px #171412 !important;
}

[data-baseweb="select"] {
  background-color: #ffffff !important;
  border: 1px solid rgba(44, 37, 34, 0.25) !important;
  border-radius: 4px !important;
}

[data-baseweb="select"] * {
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
  font-family: var(--font-sans) !important;
}

[data-testid="stSlider"] * {
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
}

div[data-testid="stThumbValue"],
div[data-testid="stTickBarMin"],
div[data-testid="stTickBarMax"] {
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] {
  background-color: #ffffff !important;
  border: 1px dashed rgba(44, 37, 34, 0.35) !important;
  border-radius: 4px !important;
}

[data-testid="stFileUploaderDropzone"] * {
  color: #2c2522 !important;
  -webkit-text-fill-color: #2c2522 !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background-color: #f9ede6 !important;
  border: 1px solid rgba(44, 37, 34, 0.2) !important;
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
}

.stCaption, [data-testid="stCaptionContainer"] p {
  color: #3d3531 !important;
  -webkit-text-fill-color: #3d3531 !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
}

.stAlert p {
  color: #171412 !important;
  -webkit-text-fill-color: #171412 !important;
}

.glyph-catalogue-item {
  background: var(--bg-white);
  border: 1px solid var(--border-editorial);
  border-radius: 4px;
  padding: 10px;
  text-align: center;
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.glyph-catalogue-item:hover {
  border-color: var(--accent-rust);
  transform: translateY(-2px);
}

.glyph-name {
  font-family: var(--font-serif);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--ink-black);
  margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_model():
    charset = ["<blank>"] + sorted(list("abcdefghijklmnopqrstuvwxyz"))
    if os.path.exists("charset.json"):
        with open("charset.json", "r", encoding="utf-8") as f:
            charset = json.load(f)
    idx2char = {i: c for i, c in enumerate(charset)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=len(charset), hidden_size=256)
    
    has_model = False
    if os.path.exists("best_crnn.pt"):
        try:
            model.load_state_dict(torch.load("best_crnn.pt", map_location=device))
            model.to(device)
            model.eval()
            has_model = True
        except Exception as e:
            print(e)
            
    return model, idx2char, device, has_model

renderer = BimaAksaraRenderer(dataset_dir="Dataset_class")
model, idx2char, device, has_model = load_ml_model()

def safe_render_aksara(renderer_inst, syllables, target_height=72, gap=10, theme="parchment"):
    try:
        return renderer_inst.render_syllables_to_image(syllables, target_height=target_height, gap=gap, theme=theme)
    except TypeError:
        return renderer_inst.render_syllables_to_image(syllables, target_height=target_height, gap=gap)

def run_crnn_prediction(img_pil):
    if not has_model:
        return None, "Model bobot 'best_crnn.pt' belum dilatih. Jalankan 'python train.py' untuk melatih model."
        
    img_gray = np.array(img_pil.convert("L"))
    h, w = img_gray.shape
    new_w = min(256, max(16, int(round(w * 32 / h))))
    resized = cv2.resize(img_gray, (new_w, 32), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((32, 256), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized
    
    tensor_img = canvas.astype(np.float32) / 255.0
    tensor_img = (tensor_img - 0.5) / 0.5
    tensor_img = torch.tensor(tensor_img).unsqueeze(0).unsqueeze(0).to(device)
    
    valid_t = max(1, new_w // 4)
    with torch.no_grad():
        log_probs = model(tensor_img)
        best_path = log_probs.argmax(dim=2).squeeze(1)[:valid_t].tolist()
        
    chars = []
    prev = 0
    for idx in best_path:
        if idx != 0 and idx != prev:
            chars.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(chars), None

st.markdown("""
<header class="editorial-topbar">
  <div style="display: flex; align-items: baseline;">
    <span class="brand-title">AKSARA BIMA</span>
    <span class="brand-badge">Filologi &amp; Naskah Nusantara</span>
  </div>
  <div class="header-colophon">
    Bo Sangaji Kai &bull; Arsip Kesultanan Bima &bull; Edisi Preservasi Digital
  </div>
</header>
<section class="hero-editorial">
  <p class="hero-manifesto-pre">Dokumentasi &amp; Preservasi Aksara Tradisional</p>
  <h1 class="hero-manifesto-title">
    Membaca Masa Lalu, <br />
    <em>Mengabadikan Aksara Bima.</em>
  </h1>
  <p class="hero-manifesto-lead">
    Sistem komprehensif kajian ortografi dan filologi bahasa Bima (Nggahi Mbojo). 
    Mengintegrasikan pemodelan visual CRNN untuk pengenalan manuskrip tulisan tangan 
    serta mesin perakit kaligrafi berbasis 90 suku kata kanonis.
  </p>
</section>
<div class="colophon-specs-grid">
  <div class="colophon-item">
    <h5>Koleksi Karakter</h5>
    <p>90 Suku Kata Kanonis (A s.d. YU)</p>
  </div>
  <div class="colophon-item">
    <h5>Basis Citra Master</h5>
    <p>2.250 Spesimen Tulisan Tangan Asli</p>
  </div>
  <div class="colophon-item">
    <h5>Mesin Inferensi</h5>
    <p>CRNN (CNN + BiLSTM + CTC) &amp; Leksikografi</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab_trans, tab_ocr, tab_katalog = st.tabs([
    "I. Studio Terjemahan & Kaligrafi Aksara",
    "II. Pembacaan Manuskrip (CRNN OCR)",
    "III. Katalog 90 Suku Kata Kanonis"
])

with tab_trans:
    st.markdown("""
    <div class="editorial-section-title">
      <span class="section-number">Bagian 1</span>
      Penggubahan Teks ke Aksara Tradisional
    </div>
    <p class="editorial-caption">
      Masukkan lema atau kalimat berbahasa Indonesia. Sistem akan mencocokkan padanan leksikon 
      Bahasa Bima, memetakan fonotaktik suku kata, dan merangkai naskah aksara fisik.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<small style='text-transform: uppercase; letter-spacing: 0.12em; color: var(--ink-light); font-weight: 600;'>Kutipan Rekomendasi:</small>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    default_text = "orang bima bicara bahasa bima"
    if c1.button("“Orang Bima”"):
        default_text = "orang bima"
    if c2.button("“Bahasa Tanah Bima”"):
        default_text = "bahasa tanah bima"
    if c3.button("“Ayah dan Ibu Makan”"):
        default_text = "ayah dan ibu makan"
    if c4.button("“Air Gunung Dingin”"):
        default_text = "air gunung dingin"

    input_teks = st.text_input("Lema / Kalimat Bahasa Indonesia:", value=default_text)

    col_opt1, col_opt2, col_opt3 = st.columns([1.2, 1, 1])
    with col_opt1:
        tema_naskah = st.selectbox(
            "Format Cetakan Naskah:",
            options=["parchment", "dark", "white", "transparent"],
            format_func=lambda x: {
                "parchment": "Kertas Lontar Kuno (Sepia & Krem Tradisional)",
                "dark": "Obsidian Editorial (Tinta Gading & Latar Gelap)",
                "white": "Lembar Naskah Putih Bersih (Standar Buku)",
                "transparent": "Matriks Transparan (PNG Tipografi)"
            }[x]
        )
    with col_opt2:
        tinggi_aksara = st.slider("Tinggi Goresan (Piksel):", min_value=52, max_value=120, value=76, step=4)
    with col_opt3:
        jarak_suku = st.slider("Jarak Spasi Antar-Aksara (Kerning):", min_value=4, max_value=24, value=10, step=2)

    if input_teks:
        bima_latin, status_tokens = translate_indonesia_to_bima(input_teks)
        
        token_markup = ""
        for t in status_tokens:
            if t["is_known"]:
                token_markup += f'<span class="word-lexicon-tag"><b>{t["source"]}</b> ➔ {t["target"]}</span>'
            else:
                token_markup += f'<span class="word-oov-tag"><b>{t["source"]}</b> ➔ {t["target"]} <small>(Serapan/Asli)</small></span>'
                
        st.markdown(f"""
        <div class="editorial-card-white">
          <h5 style='font-family: var(--font-serif); font-size: 1.15rem; margin-bottom: 0.75rem; color: var(--ink-black);'>
            1. Analisis Leksikal &amp; Morfologi
          </h5>
          <div style="margin-bottom: 0.8rem;">{token_markup}</div>
          <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--border-editorial); font-size: 1.1rem; color: var(--ink-dark);">
            Transliterasi Latin Bima: <b style="font-family: var(--font-serif); color: var(--accent-rust); font-size: 1.25rem;">“{bima_latin}”</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        words = bima_latin.split()
        all_syllables = []
        for w in words:
            syls = bima_to_syllables(w)
            all_syllables.extend(syls)
            
        syl_html = "".join([f'<span class="syllable-token">/{s}/</span>' for s in all_syllables])
        st.markdown(f"""
        <div class="editorial-card">
          <h5 style='font-family: var(--font-serif); font-size: 1.15rem; margin-bottom: 0.75rem; color: var(--ink-black);'>
            2. Pemetaan Fonotaktik 90 Suku Kata Kanonis
          </h5>
          <div>{syl_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        aksara_rendered = safe_render_aksara(
            renderer,
            all_syllables,
            target_height=tinggi_aksara,
            gap=jarak_suku,
            theme=tema_naskah
        )
        
        buf = io.BytesIO()
        aksara_rendered.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        st.markdown(f"""
        <div class="editorial-card-white">
          <h5 style="font-family: var(--font-serif); font-size: 1.15rem; margin-bottom: 0.75rem; color: var(--ink-black);">
            3. Rangkaian Naskah Aksara Bima
          </h5>
          <div class="manuscript-mat">
            <img src="data:image/png;base64,{img_b64}" alt="Aksara Bima" style="max-width: 100%; height: auto; display: block; margin: 0 auto; box-shadow: 0 4px 12px rgba(44, 37, 34, 0.08);" />
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="Simpan Lembaran Aksara (Berkas PNG)",
            data=buf.getvalue(),
            file_name=f"aksara_bima_{bima_latin.replace(' ', '_')}.png",
            mime="image/png"
        )

with tab_ocr:
    st.markdown("""
    <div class="editorial-section-title">
      <span class="section-number">Bagian 2</span>
      Transliterasi Manuskrip dengan Model CRNN
    </div>
    <p class="editorial-caption">
      Unggah potongan citra naskah bertuliskan Aksara Bima. Model jaringan saraf tiruan 
      akan memindai fitur goresan dan mendistribusikan deret huruf Latin melalui dekoder temporal CTC.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<small style='text-transform: uppercase; letter-spacing: 0.12em; color: var(--ink-light); font-weight: 600;'>Spesimen Uji Cepat:</small>", unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    sampel_terpilih = None
    
    if q1.button("Spesimen “dou”"):
        sampel_terpilih = ("dou", safe_render_aksara(renderer, ["do", "u"], target_height=64, theme="white"))
    if q2.button("Spesimen “mbojo”"):
        sampel_terpilih = ("mbojo", safe_render_aksara(renderer, ["bo", "jo"], target_height=64, theme="white"))
    if q3.button("Spesimen “nggahi”"):
        sampel_terpilih = ("nggahi", safe_render_aksara(renderer, ["ga", "hi"], target_height=64, theme="white"))
    if q4.button("Spesimen “dana”"):
        sampel_terpilih = ("dana", safe_render_aksara(renderer, ["da", "na"], target_height=64, theme="white"))
    if q5.button("Spesimen “taho”"):
        sampel_terpilih = ("taho", safe_render_aksara(renderer, ["ta", "ho"], target_height=64, theme="white"))

    col_in_img, col_out_pred = st.columns([1, 1])
    
    active_img = None
    label_sumber = ""
    
    with col_in_img:
        with st.container(border=True):
            st.markdown("<h5 style='font-family: var(--font-serif); font-size: 1.15rem; color: var(--ink-black); margin-bottom: 0.8rem;'>Berkas Manuskrip Aksara</h5>", unsafe_allow_html=True)
            unggah_berkas = st.file_uploader("Pilih berkas citra manuskrip (.jpg, .png):", type=["jpg", "png", "jpeg"])
            
            if unggah_berkas is not None:
                active_img = Image.open(unggah_berkas)
                label_sumber = unggah_berkas.name
            elif sampel_terpilih is not None:
                label_sumber, active_img = sampel_terpilih
                
            if active_img is not None:
                st.image(active_img, caption=f"Dokumen: {label_sumber}")
            else:
                st.info("Pilih salah satu spesimen uji di atas atau unggah foto goresan aksara.")
        
    with col_out_pred:
        with st.container(border=True):
            st.markdown("<h5 style='font-family: var(--font-serif); font-size: 1.15rem; color: var(--ink-black); margin-bottom: 1rem;'>Hasil Transliterasi Komputasional</h5>", unsafe_allow_html=True)
            
            if active_img is not None:
                hasil_pred, pesan_err = run_crnn_prediction(active_img)
                
                if pesan_err:
                    st.warning(pesan_err)
                    if sampel_terpilih is not None:
                        hasil_pred = sampel_terpilih[0]
                        st.markdown(f"<div style='font-size: 0.9rem; color: var(--ink-muted);'>Menampilkan rujukan kanonis: <b>{hasil_pred}</b></div>", unsafe_allow_html=True)
                        
                if hasil_pred:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-editorial); border-radius: 4px; padding: 1.5rem; text-align: center; margin-bottom: 1.2rem;">
                      <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--ink-light); margin-bottom: 0.3rem;">Bacaan Teks Latin</div>
                      <div style="font-family: var(--font-serif); font-size: 2.2rem; font-weight: 700; color: var(--accent-rust);">{hasil_pred}</div>
                      <div style="font-size: 0.8rem; color: var(--ink-muted); font-style: italic;">CTC Greedy Sequence Alignment</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    syl_pred = bima_to_syllables(hasil_pred)
                    st.markdown("**Struktur Suku Kata:**")
                    st.markdown("".join([f'<span class="syllable-token">/{s}/</span>' for s in syl_pred]), unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top: 1rem;'><b>Padanan Makna Bahasa Indonesia:</b></div>", unsafe_allow_html=True)
                    arti_list = [k for k, v in BIMA_LEXICON.items() if v == hasil_pred.lower()]
                    if arti_list:
                        st.success(f"Artian leksikal: **{', '.join(arti_list).title()}**")
                    else:
                        st.write("*(Lema belum tercatat di kamus leksikografi ringkas)*")
            else:
                st.write("Menunggu masukan berkas citra...")

with tab_katalog:
    st.markdown("""
    <div class="editorial-section-title">
      <span class="section-number">Bagian 3</span>
      Katalog Tipografi 90 Suku Kata Kanonis
    </div>
    <p class="editorial-caption">
      Daftar lengkap 90 glif suku kata Aksara Bima yang dikurasi langsung dari 2.250 spesimen naskah asli.
    </p>
    """, unsafe_allow_html=True)
    
    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        cari_suku = st.text_input("Cari Suku Kata (contoh: 'ba', 'ko', 'u'):", value="").lower().strip()
    with col_k2:
        filter_vokal = st.selectbox("Penyaring Vokal:", ["Semua Vokal", "Vokal A", "Vokal I", "Vokal U", "Vokal E", "Vokal O"])

    daftar_kelas = sorted(list(renderer.glyph_cache.keys()))
    
    terpilih = []
    for c in daftar_kelas:
        if cari_suku and cari_suku not in c:
            continue
        if filter_vokal == "Vokal A" and not c.endswith("a"):
            continue
        if filter_vokal == "Vokal I" and not c.endswith("i"):
            continue
        if filter_vokal == "Vokal U" and not c.endswith("u"):
            continue
        if filter_vokal == "Vokal E" and not c.endswith("e"):
            continue
        if filter_vokal == "Vokal O" and not c.endswith("o"):
            continue
        terpilih.append(c)

    st.markdown(f"<div style='font-size: 0.88rem; color: var(--ink-muted); margin-bottom: 1.5rem;'>Menampilkan <b>{len(terpilih)}</b> dari 90 karakter kanonis:</div>", unsafe_allow_html=True)
    
    kolom_baris = 6
    for i in range(0, len(terpilih), kolom_baris):
        slice_row = terpilih[i:i + kolom_baris]
        cols = st.columns(kolom_baris)
        for j, s in enumerate(slice_row):
            with cols[j]:
                glyph_pic = renderer.glyph_cache[s]
                st.markdown(f"""
                <div class="glyph-catalogue-item">
                  <div class="glyph-name">{s.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(glyph_pic, width='stretch')

st.markdown("""
<footer style="margin-top: 4rem; padding: 2.5rem 0; border-top: 1px solid var(--border-editorial); text-align: center;">
  <div style="font-family: var(--font-serif); font-size: 1.25rem; font-weight: 600; color: var(--ink-black); margin-bottom: 0.4rem;">
    PENERBIT AKSARA &bull; DIVISI FILOLOGI NUSANTARA
  </div>
  <p style="font-size: 0.85rem; color: var(--ink-muted); max-width: 600px; margin: 0 auto; line-height: 1.6;">
    Proyek Preservasi Digital dan Transliterasi Otomatis Naskah Kesultanan Bima (Bo Sangaji Kai). 
    Didukung oleh 2.250 data latih master suku kata dan model CRNN berstandar editorial.
  </p>
</footer>
""", unsafe_allow_html=True)
  