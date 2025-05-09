import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import json
import logging
import os
import sys
import requests
from io import BytesIO

# Tentukan BASE_DIR dan path penting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, 'Preprocessing')
STEMMER_PATH = os.path.join(BASE_DIR, 'Preprocessing', 'Stemmer', 'mpstemmer', 'mpstemmer')
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')
sys.path.append(STEMMER_PATH)
from mpstemmer import MPStemmer

# Tambahkan path untuk MPStemmer
sys.path.append(STEMMER_PATH)

# Setup NLTK data
nltk.data.path.append(NLTK_DATA_PATH)
if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers', 'punkt_tab')):
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)

# Setup logging untuk unmatched_slang.log
logging.basicConfig(filename='unmatched_slang.log', level=logging.INFO, filemode='w')

# URL API dari Hugging Face Space
API_URL = "https://johannawawi-sentiment-api.hf.space/"  # Ganti sama URL API-mu

# Fungsi untuk memanggil API
@st.cache_resource
def get_sentiment_model():
    def predict_sentiment(texts):
        if not texts:
            return [{"sentiment": "neutral", "confidence": 0.0}]
        try:
            response = requests.post(API_URL, json={"data": texts})
            if response.status_code == 200:
                results = response.json()["data"]
                return [{"sentiment": r["sentiment"], "confidence": r["confidence"]} for r in results]
            else:
                st.error(f"API error: {response.status_code}")
                return [{"sentiment": "neutral", "confidence": 0.0} for _ in texts]
        except Exception as e:
            st.error(f"Gagal memanggil API: {str(e)}")
            return [{"sentiment": "neutral", "confidence": 0.0} for _ in texts]
    return predict_sentiment

# Muat fungsi API
sentiment_model = get_sentiment_model()

# Stemmer
stemmer = MPStemmer()

# Fungsi preprocessing
def cleaningReview(review):
    if not isinstance(review, str):
        return ""
    review = re.sub(r'@\w+', ' ', review)
    review = re.sub(r'#[A-Za-z0-9]+', ' ', review)
    review = re.sub(r'http\S+', ' ', review)
    review = re.sub(r'[0-9]+', ' ', review)
    review = re.sub(r'[-()\"#/@;:<>{}\'+=~|.!?,_\*&]', ' ', review)
    review = ' '.join(review.split())
    return review

def clearEmoji(review):
    return review.encode('ascii', 'ignore').decode('ascii')

def replaceTOM(review):
    pola = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pola.sub(r'\1', review)

def casefoldingText(review):
    return review.lower()

def tokenizingText(review):
    return word_tokenize(review)

def convertToSlangword(review, kamusGabungan, debug=False):
    if not isinstance(review, list) or not review:
        return []
    text = ' '.join(str(word) for word in review if word is not None)
    SLANG_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, kamusGabungan.keys())) + r')\b', re.IGNORECASE)
    text = SLANG_PATTERN.sub(lambda x: kamusGabungan[x.group().lower()], text)
    return [word.lower() for word in text.split()]

def stemmingText(document):
    return [stemmer.stem(term) for term in document]

# Judul aplikasi
st.markdown(
    "<h1 style='text-align: center;'>🧠 Social Media Sentiment Analysis App</h1>",
    unsafe_allow_html=True
)

# Deskripsi
st.markdown(
    """
    <p style='text-align: justify;'>
        Upload your CSV or Excel file containing social media data, 
        and get sentiment insights using Natural Language Processing (NLP). 
        This application helps you understand public sentiment from textual data 
        in a simple and efficient way.
    </p>
    """,
    unsafe_allow_html=True
)

# Line
st.markdown(
    """
    <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
    """,
    unsafe_allow_html=True
)

# Unggah file dataset
uploaded_file = st.file_uploader("**📁 Upload Your Dataset to Begin**" \
                                "   \n _Only .xlsx or .csv files are supported_", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # Validasi folder Preprocessing
        if not os.path.exists(FOLDER_PATH):
            st.error(f"Preprocessing folder not found in {FOLDER_PATH}!")
            st.stop()

        # Validasi folder Stemmer
        if not os.path.exists(STEMMER_PATH):
            st.error(f"Stemmer folder not found in {STEMMER_PATH}! Pastikan folder mpstemmer ada.")
            st.stop()

        # Baca file kamus slang
        slang_file1_path = os.path.join(FOLDER_PATH, 'slangword.txt')
        slang_file2_path = os.path.join(FOLDER_PATH, 'new_kamusalay.txt')
        
        if not os.path.exists(slang_file1_path) or not os.path.exists(slang_file2_path):
            st.error(f"slangword.txt or new_kamusalay.txt not found in {FOLDER_PATH}!")
            st.stop()

        with open(slang_file1_path, "r") as f1:
            kamusSlang1 = json.load(f1)
        with open(slang_file2_path, "r") as f2:
            kamusSlang2 = json.load(f2)
        
        kamusGabungan = {k.lower(): v for k, v in {**kamusSlang1, **kamusSlang2}.items()}

        # Baca file dataset
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Tampilkan pratinjau data
        st.markdown("<h2 style='font-size: 21px;'>Dataset Preview</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin-top: -15px; font-size: 16px'>Here are the first few rows of your uploaded dataset:</p>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # Pilih kolom teks
        text_column = st.selectbox(
            "**Select the text column for analysis (e.g., full_text)**", 
            df.columns,
            index=0)
        texts = df[text_column].dropna().astype(str).tolist()

        # Preprocessing
        st.markdown("<h2 style='font-size: 21px;'>Preprocessing Process</h2>", unsafe_allow_html=True)
        with st.spinner("Processing text data..."):
            df['cleaning'] = df[text_column].apply(cleaningReview)
            df['clear_emoji'] = df['cleaning'].apply(clearEmoji)
            df['3/more'] = df['clear_emoji'].apply(replaceTOM)
            df['case_folding'] = df['3/more'].apply(casefoldingText)
            df['tokenizing'] = df['case_folding'].apply(tokenizingText)
            df['formalisasi'] = df['tokenizing'].apply(lambda x: convertToSlangword(x, kamusGabungan, debug=True))
            df['stemming'] = df['formalisasi'].apply(stemmingText)
            df['processed_text'] = df['stemming'].apply(lambda x: ' '.join(x))
        
        # Line
        st.markdown(
            """
            <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
            """,
            unsafe_allow_html=True
        )

        # Tampilkan hasil preprocessing
        st.markdown("<h2 style='font-size: 21px; margin-top: -20px'>Preprocessing Results</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin-top: -15px; font-size: 16px; margin-bottom: -15px'>Here is a preview of the preprocessed text:</p>", unsafe_allow_html=True)
        st.dataframe(df[[text_column, 'cleaning', 'case_folding', 'formalisasi', 'processed_text']].head(), use_container_width=True)

        # Line
        st.markdown(
            """
            <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
            """,
            unsafe_allow_html=True
        )

        # Analisis Sentimen
        st.markdown("<h2 style='font-size: 21px; margin-top: -20px'>Sentiment Analysis Process</h2>", unsafe_allow_html=True)
        with st.spinner("Analyzing sentiment..."):
            processed_texts = df['processed_text'].dropna().tolist()
            sentiments = sentiment_model(processed_texts)
            df['sentiment'] = [result['sentiment'] for result in sentiments]
            df['confidence'] = [result['confidence'] for result in sentiments]
            df = df.dropna(subset=['sentiment'])

        # Visualisasi
        st.markdown("<h2 style='font-size: 25px; margin-top: -5px; text-align: center; border: 1px solid grey; padding: 5px'>Visual Summary of Findings</h2>", unsafe_allow_html=True)

        # Text Box untuk Penyebaran Komentar
        st.markdown("<h4 style='text-align: center; background-color:#9EC6F3; color:black; border: 1px solid #000000; padding:1px; border-radius:10px; margin-top: 20px'>Sentiment Distribution</h4>", unsafe_allow_html=True)
        st.write("")
        sentiment_counts = df['sentiment'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            negative_count = sentiment_counts.get('negative', 0)
            st.markdown(
                f"""
                <div	td	 style="background-color:#FF8A8A; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                    <span style="font-weight: 600; font-size: 18px;">Negative</span><br>
                    <span style="font-weight: 600; font-size: 35px;">{negative_count}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            neutral_count = sentiment_counts.get('neutral', 0)
            st.markdown(
                f"""
                <div style="background-color:#F0EAAC; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                    <span style="font-weight: 600; font-size: 18px;">Neutral</span><br>
                    <span style="font-weight: 600; font-size: 35px;">{neutral_count}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            positive_count = sentiment_counts.get('positive', 0)
            st.markdown(
                f"""
                <div style="background-color:#CCE0AC; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                    <span style="font-weight: 600; font-size: 18px;">Positive</span><br>
                    <span style="font-weight: 600; font-size: 35px;">{positive_count}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Bar Chart Distribusi Sentimen
        st.markdown("<h4 style='margin-top: 20px; margin-bottom:10px; text-align: center; background-color:#9EC6F3; color:black; border: 1px solid #000000; padding:1px; border-radius:10px'>Bar Chart of Sentiment Distribution</h4>", unsafe_allow_html=True)

        order = ['negative', 'neutral', 'positive']
        sentiment_counts = sentiment_counts.reindex(order).fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        custom_colors = {
            'negative': '#FF8A8A',
            'positive': '#CCE0AC',
            'neutral': '#F0EAAC'
        }
        colors = [custom_colors.get(label, '#d3d3d3') for label in sentiment_counts.index]

        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)

        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)

        ax.set_xlabel("Sentiment", fontsize=10)
        ax.set_ylabel("Number of Comments", fontsize=10)
        ax.set_title("Sentiment Distribution", fontsize=13, pad=0)
        sns.despine()

        st.pyplot(fig)
        bar_buf = BytesIO()
        fig.savefig(bar_buf, format="png", dpi=300, bbox_inches='tight')
        bar_buf.seek(0)
        bar_col1, bar_col2, bar_col3 = st.columns([6, 1.5, 4])

        with bar_col3:
            st.download_button(
                label="📥 Download Bar Chart (PNG)",
                data=bar_buf,
                file_name="sentiment_distribution_bar_chart.png",
                mime="image/png"
            )

        # Pie Chart Sentimen
        st.markdown("<h4 style='margin-top: 20px; margin-bottom:10px; text-align: center; background-color:#9EC6F3; color:black; border: 1px solid #000000; padding:1px; border-radius:10px'>Pie Chart of Sentiment Distribution</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        explode = (0.02, 0.02, 0.02)
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=45, pctdistance=0.50,
               wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'},
               explode=explode)
        ax.set_title("Sentiment Distribution", fontsize=11, pad=0)
        st.pyplot(fig)

        pie_buf = BytesIO()
        fig.savefig(pie_buf, format="png", dpi=300, bbox_inches='tight')
        pie_buf.seek(0)

        pie_col1, pie_col2, pie_col3 = st.columns([6, 1.5, 4])

        with pie_col3:
            st.download_button(
                label="📥 Download Pie Chart (PNG)",
                data=pie_buf,
                file_name="sentiment_pie_chart.png",
                mime="image/png"
            )

        # Word Cloud untuk Sentimen
        def generate_wordcloud(text, colormap, title):
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
            fig, ax = plt.subplots(dpi=800)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=12, pad=10)
            return fig

        def get_image_download_link(fig, file_name):
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=800, bbox_inches='tight')
            buf.seek(0)
            return buf

        st.markdown("<h4 style='margin-top: 20px; margin-bottom:10px; text-align: center; background-color:#9EC6F3; color:black; border: 1px solid #000000; padding:1px; border-radius:10px'>Word Clouds of Sentiment</h4>", unsafe_allow_html=True)
        
        word_col1, word_col2 = st.columns([6, 1.5])

        with word_col1:
            positive_text = ' '.join(df[df['sentiment'] == 'positive']['processed_text'].dropna())
            if positive_text:
                fig_pos = generate_wordcloud(positive_text, 'Greens', 'Positive Sentiment WordCloud')
                st.pyplot(fig_pos)
            else:
                st.write("No positive text to display.")

            negative_text = ' '.join(df[df['sentiment'] == 'negative']['processed_text'].dropna())
            if negative_text:
                fig_neg = generate_wordcloud(negative_text, 'Reds', 'Negative Sentiment WordCloud')
                st.pyplot(fig_neg)
            else:
                st.write("No negative text to display.")
                
            neutral_text = ' '.join(df[df['sentiment'] == 'neutral']['processed_text'].dropna())
            if neutral_text:
                fig_neutral = generate_wordcloud(neutral_text, 'Purples', 'Neutral Sentiment WordCloud')
                st.pyplot(fig_neutral)
            else:
                st.write("No neutral text to display.")

        with word_col2:
            if positive_text:
                st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 150px;'>", unsafe_allow_html=True)
                buf_pos = get_image_download_link(fig_pos, "wordcloud_positive.png")
                st.download_button("📥 Download Pos_Word (HD)", buf_pos, file_name="wordcloud_positive.png", mime="image/png", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)

            if negative_text:
                st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 230px;'>", unsafe_allow_html=True)
                buf_neg = get_image_download_link(fig_neg, "wordcloud_negative.png")
                st.download_button("📥 Download Neg_Word (HD)", buf_neg, file_name="wordcloud_negative.png", mime="image/png", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)

            if neutral_text:
                st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 230px;'>", unsafe_allow_html=True)
                buf_neutral = get_image_download_link(fig_neutral, "wordcloud_neutral.png")
                st.download_button("📥 Download Neu_Word (HD)", buf_neutral, file_name="wordcloud_neutral.png", mime="image/png", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)

        # Dataset Preview with Sentiment and Confidence
        st.markdown("<h2 style='font-size: 21px; margin-top: 20px'>Final Dataset Preview</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin-top: -15px; font-size: 16px'>Preview of the original dataset with preprocessed text, sentiment, and confidence scores:</p>", unsafe_allow_html=True)
        
        preview_columns = [text_column, 'processed_text', 'sentiment', 'confidence']
        preview_df = df[preview_columns].copy()
        st.dataframe(preview_df.head(), use_container_width=True)

        csv_buf = BytesIO()
        preview_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        
        st.download_button(
            label="📥 Download Dataset with Sentiment (CSV)",
            data=csv_buf,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

        # Thank You Message
        st.markdown(
            """
            <hr style='border: 1px solid #ccc; margin-top: 20px; margin-bottom: 20px;' />
            <h3 style='text-align: center; color: #333; font-size: 24px; margin-top: 20px;'>
                Thank You for Using Our Sentiment Analysis App! 😊
            </h3>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")