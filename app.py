import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import logging
import os
from io import BytesIO
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from joblib import Parallel, delayed
from collections import Counter
import base64

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_PATH = os.path.join(BASE_DIR, 'Preprocessing')
STEMMER_PATH = os.path.join(PREPROCESSING_PATH, 'Stemmer', 'mpstemmer', 'mpstemmer')
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

# Import stemmer
try:
    from mpstemmer import MPStemmer
except ImportError:
    st.error("MPStemmer not found! Ensure the 'mpstemmer' folder exists in Preprocessing/Stemmer.")
    st.stop()

# Setup NLTK
nltk.data.path.append(NLTK_DATA_PATH)
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
try:
    if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers', 'punkt_tab')):
        nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH, quiet=True)
    if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'corpora', 'stopwords')):
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK data: {str(e)}. Try again or contact support.")
    st.stop()

try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.error("NLTK data not found. Ensure downloads succeeded and paths are correct.")
    st.stop()

# Setup logging
logging.basicConfig(filename='app_errors.log', level=logging.INFO, filemode='w')

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
        st.error(f"Failed to load model from Hugging Face: {str(e)}. Check connection or model name.")
        st.stop()

model, tokenizer, device = load_sentiment_model()

# Initialize stemmer
stemmer = MPStemmer()

# Load stopwords
indo_stopwords = set(stopwords.words('indonesian'))
custom_stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'aja', 'sih'}
indo_stopwords.update(custom_stopwords)

# Preprocessing functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9]+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\b[0-9]+\b', ' ', text)  # Remove standalone numbers
    text = re.sub(r'[-()\"#/@;:<>{}\'+=~|.!?,_\*&]', ' ', text)
    text = ' '.join(text.split())
    return text

def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def replace_repeated_chars(text):
    pattern = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pattern.sub(r'\1', text)

def lowercase_text(text):
    return text.lower()

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in indo_stopwords]

unmatched_slang = []
def convert_to_slang(text, slang_dict, debug=False):
    global unmatched_slang
    if not isinstance(text, list) or not text:
        return []
    text_str = ' '.join(str(word) for word in text if word is not None)
    SLANG_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b', re.IGNORECASE)
    text_str = SLANG_PATTERN.sub(lambda x: slang_dict[x.group().lower()], text_str)
    tokens = text_str.split()
    if debug:
        for word in tokens:
            if word.lower() not in slang_dict and word.lower() not in unmatched_slang:
                unmatched_slang.append(word.lower())
    return [word.lower() for word in tokens]

def stem_text(document):
    return [stemmer.stem(term) if len(term) > 3 else term for term in document]

# Parallel preprocessing
def preprocess_row(row, text_column, slang_dict):
    try:
        text = clean_text(row[text_column])
        text = remove_emoji(text)
        text = replace_repeated_chars(text)
        text = lowercase_text(text)
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        tokens = convert_to_slang(tokens, slang_dict, debug=True)
        tokens = stem_text(tokens)
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing row {row.name}: {str(e)}")
        return ""

# Sentiment prediction
failed_texts = []
def predict_sentiment(texts, batch_size=16):
    global failed_texts
    if not texts:
        return [{"sentiment": "neutral", "confidence": 0.0} for _ in texts]
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = []
    
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
            logging.error(f"Error in batch {i}: {str(e)}")
            failed_texts.extend(batch_texts)
            results.extend([{"sentiment": "neutral", "confidence": 0.0}] * len(batch_texts))
        progress = min((i + len(batch_texts)) / len(texts), 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return results

# Top keywords per sentiment
def top_keywords(df, sentiment, n=5):
    texts = df[df['sentiment_result'] == sentiment]['processed_text'].str.split()
    words = [word for text in texts for word in text if word]
    return Counter(words).most_common(n)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stSelectbox { margin-bottom: 20px; }
    .stSpinner { margin: 20px 0; }
    .header { text-align: center; color: #333; font-size: 28px; margin-bottom: 10px; }
    .subheader { text-align: center; color: #555; font-size: 18px; margin-bottom: 20px; }
    .divider { border: 1px solid #ccc; margin: 20px 0; }
    .sentiment-card { padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
    <div class='header'>🧠 Social Media Sentiment Analyzer</div>
    <div class='subheader'>Upload your data and vibe-check it with dope NLP! 🚀</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Your Dataset (CSV/XLSX)",
    type=["csv", "xlsx"],
    help="Upload a file with a text column (e.g., social media comments)."
)

if uploaded_file is not None:
    try:
        # Validate paths
        if not os.path.exists(PREPROCESSING_PATH):
            st.error(f"Preprocessing folder not found at {PREPROCESSING_PATH}!")
            st.stop()
        if not os.path.exists(STEMMER_PATH):
            st.error(f"MPStemmer folder not found at {STEMMER_PATH}!")
            st.stop()

        # Load slang dictionaries
        slang_file1_path = os.path.join(PREPROCESSING_PATH, 'slangword.txt')
        slang_file2_path = os.path.join(PREPROCESSING_PATH, 'new_kamusalay.txt')
        slang_dict1, slang_dict2 = {}, {}
        for path, dict_name in [(slang_file1_path, 'slangword.txt'), (slang_file2_path, 'new_kamusalay.txt')]:
            if not os.path.exists(path):
                st.error(f"File {dict_name} not found at {PREPROCESSING_PATH}!")
                st.stop()
            try:
                with open(path, "r", encoding='utf-8') as f:
                    if path == slang_file1_path:
                        slang_dict1 = json.load(f)
                    else:
                        slang_dict2 = json.load(f)
            except json.JSONDecodeError:
                st.error(f"File {dict_name} is not valid JSON!")
                st.stop()
            except Exception as e:
                st.error(f"Failed to load file {dict_name}: {str(e)}")
                st.stop()
        combined_slang_dict = {k.lower(): v for k, v in {**slang_dict1, **slang_dict2}.items()}

        # Read dataset
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        original_columns = list(df.columns)

        # Filter text columns
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 10]
        if not text_columns:
            st.error("No valid text columns found in the dataset. Choose a file with text data (e.g., comments).")
            st.stop()
        
        # Select text column and max rows
        col1, col2 = st.columns([3, 1])
        with col1:
            text_column = st.selectbox(
                "Select the text column for analysis",
                text_columns,
                help="Choose a column with text, like 'full_text'."
            )
        with col2:
            max_rows = st.number_input(
                "Max rows to analyze",
                min_value=1,
                max_value=len(df),
                value=min(1000, len(df)),
                help="Limit rows to speed things up."
            )
        
        # Filter dataset
        df = df.head(max_rows).copy()
        texts = df[text_column].dropna().astype(str).tolist()
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            st.error(f"Column '{text_column}' has no valid text. Pick another column!")
            st.stop()

        # Preprocessing
        with st.spinner("Processing text, hang tight bro..."):
            progress_bar = st.progress(0)
            df['processed_text'] = Parallel(n_jobs=-1)(
                delayed(preprocess_row)(row, text_column, combined_slang_dict) for _, row in df.iterrows()
            )
            original_row_count = len(df)
            df = df[df['processed_text'].str.strip().astype(bool)].copy()
            removed_rows = original_row_count - len(df)
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} rows with empty text after preprocessing.")
            if df.empty:
                st.error("No valid text left after preprocessing. Check your dataset or preprocessing steps!")
                st.stop()
            progress_bar.progress(1.0)
            progress_bar.empty()

        # Sentiment analysis
        with st.spinner("Analyzing sentiments, almost there..."):
            progress_bar = st.progress(0)
            processed_texts = df['processed_text'].dropna().tolist()
            sentiments = predict_sentiment(processed_texts)
            df['sentiment_result'] = [result['sentiment'] for result in sentiments]
            df['confidence'] = [result['confidence'] for result in sentiments]
            df = df.dropna(subset=['sentiment_result'])
            if failed_texts:
                st.warning(f"Failed to analyze {len(failed_texts)} texts. Check app_errors.log for details.")
            progress_bar.empty()

        # Sentiment filter
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
            help="Choose which sentiments to display."
        )
        filtered_df = df[df['sentiment_result'].isin(sentiment_filter)].copy()
        if filtered_df.empty:
            st.warning("No data matches the sentiment filter. Try different sentiments!")
            filtered_df = df.copy()

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data & Preprocessing", "📈 Visualizations", "📥 Download"])

        # Tab 1: Data Preview and Preprocessing
        with tab1:
            st.markdown("<h2>Original Data</h2>", unsafe_allow_html=True)
            st.dataframe(df[original_columns], use_container_width=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<h2>Preprocessed Data</h2>", unsafe_allow_html=True)
            st.dataframe(filtered_df[[text_column, 'processed_text', 'sentiment_result', 'confidence']], use_container_width=True)
            if unmatched_slang:
                st.warning(f"Found {len(unmatched_slang)} unmatched slang words. Check app_errors.log.")
                logging.info(f"Unmatched slang: {unmatched_slang[:50]}")

        # Tab 2: Visualizations
        with tab2:
            st.markdown("<h2>Visual Summary</h2>", unsafe_allow_html=True)
            
            # Sentiment cards
            sentiment_counts = filtered_df['sentiment_result'].value_counts()
            col1, col2, col3 = st.columns(3)
            for col, sentiment, color, emoji in [
                (col1, 'negative', '#FF8A8A', '☹️'),
                (col2, 'neutral', '#F0EAAC', '😐'),
                (col3, 'positive', '#CCE0AC', '☺️')
            ]:
                count = sentiment_counts.get(sentiment, 0)
                with col:
                    st.markdown(
                        f"""
                        <div class='sentiment-card' style='background-color:{color}; border:1px solid #000;'>
                            <span style='font-size:18px; font-weight:600;'>{sentiment.capitalize()} {emoji}</span><br>
                            <span style='font-size:35px; font-weight:600;'>{count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Bar chart
            st.markdown("<h3>Sentiment Distribution (Bar)</h3>", unsafe_allow_html=True)
            order = ['negative', 'neutral', 'positive']
            sentiment_counts = sentiment_counts.reindex(order).fillna(0).reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            colors = ['#FF8A8A', '#F0EAAC', '#CCE0AC']
            fig_bar = px.bar(
                sentiment_counts,
                x='sentiment',
                y='count',
                color='sentiment',
                color_discrete_sequence=colors,
                text='count'
            )
            fig_bar.update_traces(
                textposition='outside',
                hovertemplate='Sentiment: %{x}<br>Count: %{y}<br>Percentage: %{y/sum(sentiment_counts["count"])*100:.1f}%'
            )
            fig_bar.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Count",
                showlegend=False,
                height=500,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            bar_buf = BytesIO()
            fig_bar.write_image(bar_buf, format="png", scale=3)
            st.download_button(
                label="📥 Download Bar Chart",
                data=bar_buf,
                file_name="sentiment_bar.png",
                mime="image/png"
            )

            # Pie chart
            st.markdown("<h3>Sentiment Distribution (Pie)</h3>", unsafe_allow_html=True)
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=sentiment_counts['sentiment'],
                    values=sentiment_counts['count'],
                    textinfo='percent+label',
                    marker=dict(colors=colors, line=dict(color='black', width=1)),
                    pull=[0.02, 0.02, 0.02]
                )
            ])
            fig_pie.update_layout(
                showlegend=True,
                height=500,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            pie_buf = BytesIO()
            fig_pie.write_image(pie_buf, format="png", scale=3)
            st.download_button(
                label="📥 Download Pie Chart",
                data=pie_buf,
                file_name="sentiment_pie.png",
                mime="image/png"
            )

            # Confidence score distribution
            st.markdown("<h3>Confidence Score Distribution</h3>", unsafe_allow_html=True)
            fig_conf = px.histogram(
                filtered_df,
                x='confidence',
                nbins=20,
                color='sentiment_result',
                color_discrete_sequence=colors,
                title="Confidence Score Distribution by Sentiment"
            )
            fig_conf.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                height=500,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            conf_buf = BytesIO()
            fig_conf.write_image(conf_buf, format="png", scale=3)
            st.download_button(
                label="📥 Download Confidence Histogram",
                data=conf_buf,
                file_name="confidence_histogram.png",
                mime="image/png"
            )

            # Word clouds
            st.markdown("<h3>Word Cloud by Sentiment</h3>", unsafe_allow_html=True)
            def generate_wordcloud(text, colormap, title):
                if not text.strip():
                    return None
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
                fig, ax = plt.subplots(dpi=800)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(title, fontsize=12, pad=10)
                return fig

            sentiments_available = {}
            for sentiment in ['positive', 'negative', 'neutral']:
                text = ' '.join(filtered_df[filtered_df['sentiment_result'] == sentiment]['processed_text'].dropna())
                sentiments_available[sentiment] = bool(text.strip())

            if not any(sentiments_available.values()):
                st.warning("No text available for word clouds. Check your sentiment filter!")
            else:
                for sentiment, colormap in [('positive', 'Greens'), ('negative', 'Reds'), ('neutral', 'Purples')]:
                    if sentiments_available[sentiment]:
                        text = ' '.join(filtered_df[filtered_df['sentiment_result'] == sentiment]['processed_text'].dropna())
                        fig = generate_wordcloud(text, colormap, f"{sentiment.capitalize()} Word Cloud")
                        st.pyplot(fig)
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=800, bbox_inches='tight')
                        st.download_button(
                            label=f"📥 Download {sentiment.capitalize()} Word Cloud",
                            data=buf,
                            file_name=f"wordcloud_{sentiment}.png",
                            mime="image/png"
                        )

            # Top keywords
            st.markdown("<h3>Top Keywords by Sentiment</h3>", unsafe_allow_html=True)
            for sentiment in ['positive', 'negative', 'neutral']:
                keywords = top_keywords(filtered_df, sentiment)
                if keywords:
                    st.write(f"**{sentiment.capitalize()}**: {', '.join([f'{k} ({v})' for k, v in keywords])}")

        # Tab 3: Download Results
        with tab3:
            st.markdown("<h2>Final Dataset</h2>", unsafe_allow_html=True)
            preview_columns = original_columns + ['processed_text', 'sentiment_result', 'confidence']
            preview_columns = [col for col in preview_columns if col in filtered_df.columns]
            st.dataframe(filtered_df[preview_columns], use_container_width=True)
            csv_buf = BytesIO()
            filtered_df.to_csv(csv_buf, index=False, sep=';')
            st.download_button(
                label="📥 Download Dataset with Sentiments",
                data=csv_buf,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

        # Feedback form
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2>Drop Your Feedback!</h2>", unsafe_allow_html=True)
        feedback = st.text_area("What do you think about this app? 😎")
        if st.button("Send Feedback"):
            logging.info(f"User feedback: {feedback}")
            st.success("Yo, thanks for the feedback, bro!")

    except Exception as e:
        st.error(f"Oops, something broke: {str(e)}. Check app_errors.log or ping support.")
        logging.exception("Main error")
        st.stop()

# Thank you message
st.markdown("""
    <div class='divider'></div>
    <div class='subheader'>Thanks for using Social Media Sentiment Analyzer! You're awesome! 😎</div>
""", unsafe_allow_html=True)