import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import logging
import os
from io import BytesIO
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from mpstemmer import MPStemmer

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_PATH = os.path.join(BASE_DIR, 'Preprocessing')
STEMMER_PATH = os.path.join(PREPROCESSING_PATH, 'Stemmer', 'mpstemmer', 'mpstemmer')
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

# Setup NLTK data path
nltk.data.path.append(NLTK_DATA_PATH)
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

# Download required NLTK data
try:
    for resource in ['punkt_tab', 'stopwords']:
        if not os.path.exists(os.path.join(NLTK_DATA_PATH, resource)):
            nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resource: {str(e)}")
    st.stop()

# Setup logging
logging.basicConfig(filename='sentiment_analysis.log', level=logging.INFO, filemode='w')

# Custom CSS for improved UI
st.markdown("""
    <style>
    body { font-family: 'Inter', sans-serif; }
    h2 { font-size: 24px; font-weight: 600; margin-bottom: 10px; }
    h4 { font-size: 18px; font-weight: 500; margin-bottom: 8px; }
    p { font-size: 16px; line-height: 1.5; }
    .stDownloadButton > button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #357ABD;
        transform: scale(1.05);
    }
    hr { margin: 20px 0; border: 1px solid #ccc; }
    @media (max-width: 768px) {
        .stColumn { flex-direction: column; align-items: center; gap: 10px; }
        .stDownloadButton { width: 100%; }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "johannawawi/v3_balanced_dataset_fine-tuning-java-indo-sentiment-analysist-3-class"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model, tokenizer, device = load_sentiment_model()

# Initialize stemmer
stemmer = MPStemmer()

# Preprocessing functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9]+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[-()\"#/@;:<>{}\'+=~|.!?,_\*&]%', ' ', text)
    return ' '.join(text.split())

def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def replace_repeated_chars(text):
    pattern = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pattern.sub(r'\1', text)

def lowercase_text(text):
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)

@st.cache_data
def convert_to_slang(text, slang_dict):
    if not isinstance(text, list) or not text:
        return []
    text_str = ' '.join(str(word) for word in text if word is not None)
    SLANG_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b', re.IGNORECASE)
    text_str = SLANG_PATTERN.sub(lambda x: slang_dict[x.group().lower()], text_str)
    return [word.lower() for word in text_str.split()]

def remove_stopwords(tokens):
    stopwords_id = set(stopwords.words('indonesian')).union({'dan', 'di', 'ke', 'dari', 'yang'})
    return [word for word in tokens if word not in stopwords_id]

def stem_text(document):
    return [stemmer.stem(term) for term in document]

# Sentiment prediction
def predict_sentiment(texts, batch_size=32 if torch.cuda.is_available() else 8):
    if not texts:
        return [{"sentiment": "neutral", "confidence": 0.0} for _ in texts]
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = []
    
    progress_bar = st.progress(0)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                for pred, prob in zip(predictions, probabilities):
                    label = label_map[pred.item()]
                    confidence = prob[pred].item()
                    results.append({"sentiment": label, "confidence": confidence})
        except Exception as e:
            st.warning(f"Error in sentiment analysis: {str(e)}")
            logging.error(f"Sentiment error in batch {i}: {str(e)}")
            results.extend([{"sentiment": "neutral", "confidence": 0.0}] * len(batch_texts))
        
        progress_bar.progress(min((i + len(batch_texts)) / len(texts), 1.0))
    
    progress_bar.empty()
    return results

# App title and description
st.markdown("""
    <h1 style='text-align: center;'>🧠 Social Media Sentiment Analysis</h1>
    <p style='text-align: justify;'>
        Upload a CSV or Excel file with social media text data to analyze sentiments using advanced NLP techniques. 
        Explore preprocessing results, visualizations, and download insights with ease.
    </p>
    <hr>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload Your Dataset (CSV or Excel)",
    type=["xlsx", "csv"],
    help="Supported formats: .csv, .xlsx. Ensure your file contains a column with text data."
)

if uploaded_file:
    try:
        # Validate paths
        for path, name in [(PREPROCESSING_PATH, "Preprocessing folder"), (STEMMER_PATH, "Stemmer folder")]:
            if not os.path.exists(path):
                st.error(f"{name} not found at {path}!")
                st.stop()

        # Load slang dictionaries
        slang_file1_path = os.path.join(PREPROCESSING_PATH, 'slangword.txt')
        slang_file2_path = os.path.join(PREPROCESSING_PATH, 'new_kamusalay.txt')
        for path in [slang_file1_path, slang_file2_path]:
            if not os.path.exists(path):
                st.error(f"Slang file not found at {path}!")
                st.stop()

        try:
            with open(slang_file1_path, "r", encoding='utf-8') as f1, open(slang_file2_path, "r", encoding='utf-8') as f2:
                slang_dict1 = json.load(f1)
                slang_dict2 = json.load(f2)
            combined_slang_dict = {k.lower(): v for k, v in {**slang_dict1, **slang_dict2}.items()}
        except Exception as e:
            st.error(f"Error loading slang dictionaries: {str(e)}")
            st.stop()

        # Optional custom slang dictionary
        slang_file = st.file_uploader("Upload Custom Slang Dictionary (JSON, optional)", type=["json"])
        if slang_file:
            try:
                custom_slang = json.load(slang_file)
                combined_slang_dict.update({k.lower(): v for k, v in custom_slang.items()})
                st.success("Custom slang dictionary loaded successfully!")
            except Exception as e:
                st.warning(f"Error loading custom slang: {str(e)}")

        # Load dataset
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        original_columns = list(df.columns)

        # Text column selection
        text_column = st.selectbox(
            "Select Text Column for Analysis",
            df.columns,
            help="Choose the column containing text data (e.g., tweets, comments)."
        )
        texts = df[text_column].dropna().astype(str).tolist()
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            st.error(f"No valid text data in column '{text_column}'. Please select a column with text content.")
            st.stop()

        # Preprocessing
        with st.spinner("Preprocessing text data..."):
            progress_bar = st.progress(0)
            steps = 8
            preprocessing_steps = [
                ("Cleaning text", clean_text, 'cleaned_text'),
                ("Removing emojis", remove_emoji, 'emoji_removed'),
                ("Normalizing characters", replace_repeated_chars, 'repeated_chars_removed'),
                ("Lowercasing", lowercase_text, 'lowercased'),
                ("Tokenizing", tokenize_text, 'tokenized'),
                ("Converting slang", lambda x: convert_to_slang(x, combined_slang_dict), 'slang_converted'),
                ("Removing stopwords", remove_stopwords, 'stopwords_removed'),
                ("Stemming", stem_text, 'stemmed')
            ]
            for i, (desc, func, col) in enumerate(preprocessing_steps):
                df[col] = df[preprocessing_steps[i-1][2] if i > 0 else text_column].apply(func)
                progress_bar.progress((i + 1) / steps, text=f"{desc}...")
            
            df['processed_text'] = df['stemmed'].apply(lambda x: ' '.join(x) if x else '')
            original_row_count = len(df)
            df = df[df['processed_text'].str.strip().astype(bool)].copy()
            if len(df) < original_row_count:
                st.warning(f"Removed {original_row_count - len(df)} rows with empty text after preprocessing.")
            if df.empty:
                st.error("No valid text remains after preprocessing. Please check your data.")
                st.stop()
            progress_bar.empty()

        # Sentiment analysis
        with st.spinner("Analyzing sentiments..."):
            processed_texts = df['processed_text'].tolist()
            sentiments = predict_sentiment(processed_texts)
            df['sentiment_result'] = [result['sentiment'] for result in sentiments]
            df['confidence'] = [result['confidence'] for result in sentiments]
            df['low_confidence'] = df['confidence'] < 0.5
            if df['low_confidence'].any():
                st.warning(f"{df['low_confidence'].sum()} predictions have low confidence (<0.5).")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Visualizations", "📥 Download Results"])

        # Tab 1: Data Preview
        with tab1:
            st.markdown("<h2>Dataset Preview</h2><p>View the first few rows of your uploaded dataset.</p>", unsafe_allow_html=True)
            st.dataframe(df[original_columns], use_container_width=True, height=300)
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander("Show Preprocessing Details"):
                st.markdown("<h2>Preprocessing Results</h2><p>Preview the text after each preprocessing step.</p>", unsafe_allow_html=True)
                st.dataframe(df[[text_column, 'cleaned_text', 'lowercased', 'slang_converted', 'processed_text']])

        # Tab 2: Visualizations
        with tab2:
            st.markdown("<h2 style='text-align: center;'>Visual Insights</h2>", unsafe_allow_html=True)
            sentiment_counts = df['sentiment_result'].value_counts()
            col1, col2, col3 = st.columns(3)
            for col, sent, color, emoji in [
                (col1, 'negative', '#FF8A8A', '☹️'),
                (col2, 'neutral', '#F0EAAC', '😐'),
                (col3, 'positive', '#CCE0AC', '☺️')
            ]:
                count = sentiment_counts.get(sent, 0)
                col.markdown(f"""
                    <div style='background-color:{color}; color:black; border:1px solid #000; padding:10px; border-radius:10px; text-align:center;'>
                        <span style='font-weight:600; font-size:18px;'>{sent.capitalize()} {emoji}</span><br>
                        <span style='font-weight:600; font-size:35px;'>{count}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Bar chart
            st.markdown("<h4 style='text-align:center; background-color:#9EC6F3; border:1px solid #000; padding:5px; border-radius:10px;'>Sentiment Distribution (Bar)</h4>", unsafe_allow_html=True)
            order = ['negative', 'neutral', 'positive']
            sentiment_counts_df = sentiment_counts.reindex(order).fillna(0).reset_index()
            sentiment_counts_df.columns = ['sentiment', 'count']
            colors = {'negative': '#FF8A8A', 'neutral': '#F0EAAC', 'positive': '#CCE0AC'}
            fig_bar = px.bar(
                sentiment_counts_df,
                x='sentiment',
                y='count',
                color='sentiment',
                color_discrete_map=colors,
                text='count'
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                xaxis_title="Sentiment", yaxis_title="Count", showlegend=False,
                height=500, plot_bgcolor='white', autosize=True
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            bar_buf = BytesIO()
            fig_bar.write_image(bar_buf, format="png", scale=3)
            _, col, _ = st.columns([6, 1.5, 4])
            col.download_button(
                label="Download Bar Chart",
                data=bar_buf,
                file_name="sentiment_bar.png",
                mime="image/png",
                help="Download the bar chart as a PNG image."
            )

            # Pie chart
            st.markdown("<h4 style='text-align:center; background-color:#9EC6F3; border:1px solid #000; padding:5px; border-radius:10px;'>Sentiment Distribution (Pie)</h4>", unsafe_allow_html=True)
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=sentiment_counts_df['sentiment'],
                    values=sentiment_counts_df['count'],
                    textinfo='percent+label',
                    marker=dict(colors=[colors[s] for s in sentiment_counts_df['sentiment']], line=dict(color='black', width=1)),
                    pull=[0.02] * 3
                )
            ])
            fig_pie.update_layout(height=500, showlegend=True, plot_bgcolor='white', autosize=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            pie_buf = BytesIO()
            fig_pie.write_image(pie_buf, format="png", scale=3)
            _, col, _ = st.columns([6, 1.5, 4])
            col.download_button(
                label="Download Pie Chart",
                data=pie_buf,
                file_name="sentiment_pie.png",
                mime="image/png"
            )

            # Word clouds
            st.markdown("<h4 style='text-align:center; background-color:#9EC6F3; border:1px solid #000; padding:5px; border-radius:10px;'>Word Clouds</h4>", unsafe_allow_html=True)
            sentiment_filter = st.multiselect(
                "Select Sentiments for Word Clouds",
                ['positive', 'negative', 'neutral'],
                default=['positive', 'negative', 'neutral']
            )
            colormap_dict = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Purples'}
            for sentiment in sentiment_filter:
                text = ' '.join(df[df['sentiment_result'] == sentiment]['processed_text'].dropna())
                if text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap_dict[sentiment], stopwords=stopwords.words('indonesian')).generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f"{sentiment.capitalize()} Word Cloud")
                    st.pyplot(fig)
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    col1, col2, col3 = st.columns(3)
                    col2.download_button(
                        label=f"Download {sentiment.capitalize()} Word Cloud",
                        data=buf,
                        file_name=f"wordcloud_{sentiment}.png",
                        mime="image/png"
                    )

        # Tab 3: Download
        with tab3:
            st.markdown("<h2>Download Results</h2><p>Preview and download the dataset with sentiment analysis results.</p>", unsafe_allow_html=True)
            preview_columns = original_columns + ['processed_text', 'sentiment_result', 'confidence', 'low_confidence']
            preview_columns = [col for col in preview_columns if col in df.columns]
            st.dataframe(df[preview_columns], use_container_width=True, height=300)
            csv_buf = BytesIO()
            df[preview_columns].to_csv(csv_buf, index=False, sep=';')
            st.download_button(
                label="Download CSV",
                data=csv_buf,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
            excel_buf = BytesIO()
            df[preview_columns].to_excel(excel_buf, index=False)
            st.download_button(
                label="Download Excel",
                data=excel_buf,
                file_name="sentiment_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.markdown("<hr><h3 style='text-align: center;'>Thank You for Using Our App! 😊</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("Main processing error")
        st.stop()