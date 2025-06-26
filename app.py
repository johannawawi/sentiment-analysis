# Import Libraries
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
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

# Configuration and Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_PATH = os.path.join(BASE_DIR, 'preprocessing')
STEMMER_PATH = os.path.join(PREPROCESSING_PATH, 'stemmer', 'mpstemmer', 'mpstemmer')
STOPWORD_PATH = os.path.join(PREPROCESSING_PATH, 'stopword')
SLANGWORD_PATH = os.path.join(PREPROCESSING_PATH, 'slangword')

# Define paths for 3 stopword and 3 slangword files
STOPWORD_ID_PATH = os.path.join(STOPWORD_PATH, 'stopword_id.txt')
STOPWORD_JV_PATH = os.path.join(STOPWORD_PATH, 'stopword_jv.txt')
STOPWORD_CUSTOM_PATH = os.path.join(STOPWORD_PATH, 'stopword_sby.txt')
SLANG_FILE1_PATH = os.path.join(SLANGWORD_PATH, 'slangword.txt')
SLANG_FILE2_PATH = os.path.join(SLANGWORD_PATH, 'new_kamusalay.txt')
SLANG_FILE3_PATH = os.path.join(SLANGWORD_PATH, 'slangword_sby.txt')

NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

from mpstemmer import MPStemmer

# Setup Logging
logging.basicConfig(filename='unmatched_slang.log', level=logging.INFO, filemode='w')

st.set_page_config(layout="centered")

# Initialize NLTK
def initialize_nltk():
    """Setup NLTK data and ensure punkt_tab is available."""
    nltk.data.path.append(NLTK_DATA_PATH)
    if not os.path.exists(NLTK_DATA_PATH):
        os.makedirs(NLTK_DATA_PATH)
    try:
        if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers', 'punkt_tab')):
            nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH, quiet=True)
        nltk.data.find('tokenizers/punkt_tab')
    except Exception as e:
        st.error(f"Failed to setup NLTK data: {str(e)}")
        st.stop()

# Load Resources
@st.cache_resource
def load_sentiment_model():
    """Load Hugging Face sentiment model and tokenizer."""
    try:
        model_name = "johannawawi/model-for-sosmed-analysis-v3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load sentiment model: {str(e)}")
        st.stop()

@st.cache_resource
def load_slang_dictionary():
    """Load and combine three slang dictionaries."""
    try:
        slang_paths = [SLANG_FILE1_PATH, SLANG_FILE2_PATH, SLANG_FILE3_PATH]
        for path in slang_paths:
            if not os.path.exists(path):
                st.error(f"Slang file not found at {path}")
                st.stop()
        slang_dict = {}
        for path in slang_paths:
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    slang_dict.update({k.lower(): v for k, v in json.load(f).items()})
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON format in slang file: {path}")
                    st.stop()
        return slang_dict
    except Exception as e:
        st.error(f"Failed to load slang dictionaries: {str(e)}")
        st.stop()

@st.cache_resource
def load_custom_stopwords():
    """Load and combine three stopword files."""
    try:
        stopword_paths = [STOPWORD_ID_PATH, STOPWORD_JV_PATH, STOPWORD_CUSTOM_PATH]
        for path in stopword_paths:
            if not os.path.exists(path):
                st.error(f"Stopword file not found at {path}")
                st.stop()
        stopwords = set()
        for path in stopword_paths:
            with open(path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip().lower() for line in f if line.strip())
        return stopwords
    except Exception as e:
        st.error(f"Failed to load stopwords: {str(e)}")
        st.stop()

# Preprocessing Functions
def clean_text(text):
    """Clean text by removing mentions, hashtags, URLs, numbers, and special characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9]+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[-()\"#/@;:<>{}\'+=~|.!?,_\*&]', ' ', text)
    return ' '.join(text.split())

def remove_emoji(text):
    """Remove emojis from text."""
    return text.encode('ascii', 'ignore').decode('ascii')

def replace_repeated_chars(text):
    """Replace three or more repeated characters with a single character."""
    return re.sub(r'(.)\1{2,}', r'\1', text)

def lowercase_text(text):
    """Convert text to lowercase."""
    return text.lower()

def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)

def convert_to_slang(tokens, slang_dict):
    """Convert slang words using the provided dictionary."""
    if not isinstance(tokens, list) or not tokens:
        return []
    text = ' '.join(str(word) for word in tokens if word is not None)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b', re.IGNORECASE)
    text = pattern.sub(lambda x: slang_dict[x.group().lower()], text)
    return text.lower().split()

def remove_stopwords(tokens, stopwords):
    """Remove stopwords from token list."""
    return [word for word in tokens if word.lower() not in stopwords]

def stem_text(tokens):
    """Stem tokens using MPStemmer."""
    stemmer = MPStemmer()
    return [stemmer.stem(term) for term in tokens]

# Sentiment Analysis
def predict_sentiment(texts, model, tokenizer, device, batch_size=16):
    """Predict sentiment for a list of texts."""
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
                    results.append({
                        "sentiment": label_map[pred.item()],
                        "confidence": prob[pred].item()
                    })
        except Exception as e:
            st.warning(f"Sentiment analysis failed for batch: {str(e)}")
            results.extend([{"sentiment": "neutral", "confidence": 0.0} for _ in batch_texts])
        progress_bar.progress(min((i + len(batch_texts)) / len(texts), 1.0))
    
    progress_bar.empty()
    return results

# Visualization Functions
def generate_wordcloud(text, colormap, title):
    """Generate a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=10)
    plt.tight_layout()
    return fig

def get_image_buffer(fig):
    """Convert matplotlib figure to PNG buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=800, bbox_inches='tight')
    buf.seek(0)
    return buf

# Custom CSS
def apply_custom_css():
    """Apply custom CSS for improved UI."""
    st.markdown("""
        <style>
        .stApp {
            background-color: #F4F4F4;
            background-image: url("http://www.transparenttextures.com/patterns/beige-paper.png");
            background-size: auto;
            background-repeat: repeat;
        }
        h1 { font-size: 32px; font-weight: 700; font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333;}
        h2 { font-size: 24px; font-weight: 600; font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333;}
        h3 { font-size: 21px; font-weight: 600; font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333;}
        h4 { font-size: 16px; font-weight: 600; font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333;}
        p { font-size: 14px; line-height: 1.5; font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333;}
        span { font-weight: 600; }
        .stTabs { 
            background: linear-gradient(to bottom, #f8f9fa, #e9ecef); 
            padding: 12px; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky; 
            top: 0; 
            z-index: 100;
        }
        </style>
    """, unsafe_allow_html=True)

# Thank you message
thank_you_message = """
<hr style='border: 1px solid #ccc; margin: 20px 0;' />
<h3 style='text-align: center;'>
    Thank You for Using Our Sentiment Analysis App! 😊
</h3>
"""

# Main Application
def main():
    """Main Streamlit application."""
    apply_custom_css()
    
    # Title and Description
    st.markdown("""
        <div style="background: linear-gradient(to bottom, #f8f9fa, #e9ecef); padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin: 20px 0;">
            <h1 style="font-size: 32px; font-weight: 700; color: #333; margin-bottom: 10px;">
                🧠 Social Media Sentiment Analysis App
            </h1>
            <p style="font-size: 18px; line-height: 1.5; color: #333; margin: 0 auto; max-width: 700px; text-align: justify">
                Upload your CSV or Excel file containing social media data, 
                and get sentiment insights using Natural Language Processing (NLP). 
                This app helps you understand public sentiment from text data in a simple and efficient way.
            </p>
        </div>
        <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
    """, unsafe_allow_html=True)

    # File Uploader
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
        
    uploaded_file = st.file_uploader(
            "**📁 Upload Your Dataset to Start**   \n\n Only .xlsx or .csv files are supported",
            type=["xlsx", "csv"],
            help="Upload an Excel (.xlsx) or CSV (.csv) file containing your dataset. Click the 'X' to clear the current file or 'Browse files' to upload a new one.",
            key="file_uploader"
        )

    # Update session state based on uploader
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.markdown(
            "<p style='font-size: 14px; color: #555;'>Click the 'X' next to the file name to clear it, or use 'Browse files' to upload a new file.</p>",
            unsafe_allow_html=True
        )
        st.success(f"File '{uploaded_file.name}' successfully uploaded!", icon="✅")

    else:
        st.session_state.uploaded_file = None
        st.info("Please upload a .xlsx or .csv file to start.")

    # Process file if available
    if st.session_state.uploaded_file is not None:
        try:
            if st.session_state.uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(st.session_state.uploaded_file)
            else:
                try:
                    df = pd.read_csv(st.session_state.uploaded_file, sep=';')
                    if len(df.columns) <= 1:
                        st.session_state.uploaded_file.seek(0)
                        df = pd.read_csv(st.session_state.uploaded_file, sep=',')
                except:
                    st.session_state.uploaded_file.seek(0)
                    df = pd.read_csv(st.session_state.uploaded_file, sep=',')

            original_columns = list(df.columns)
            # Initialize Resources
            initialize_nltk()
            model, tokenizer, device = load_sentiment_model()
            slang_dict = load_slang_dictionary()
            custom_stopwords = load_custom_stopwords()
    
            # Select Text Column
            text_column = st.selectbox(
                "**Select the text column for analysis**",
                df.columns,
                index=0,
                help="Choose the column containing the main text (e.g., reviews, tweets)."
            )

            original_row_count = len(df)
            df = df.dropna(subset=[text_column])
            df[text_column] = df[text_column].astype(str).str.strip()
            df = df[df[text_column].astype(bool)].drop_duplicates(subset=[text_column], keep='first')
            texts = df[text_column].tolist()
            if not [t for t in texts if t.strip()]:
                st.error(f"No valid text data in column '{text_column}'. Please select a different column.")
                st.stop()
    
            # Preprocessing
            steps = 9
            step = 0
            
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
        
            with st.spinner("Cleaning text..."):
                df['cleaned_text'] = df[text_column].apply(clean_text)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Removing emojis..."):
                df['emoji_removed'] = df['cleaned_text'].apply(remove_emoji)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Removing repeated characters..."):
                df['repeated_chars_removed'] = df['emoji_removed'].apply(replace_repeated_chars)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Converting to lowercase..."):
                df['lowercased'] = df['repeated_chars_removed'].apply(lowercase_text)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Tokenizing text..."):
                df['tokenized'] = df['lowercased'].apply(tokenize_text)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Converting slang..."):
                df['slang_converted'] = df['tokenized'].apply(lambda x: convert_to_slang(x, slang_dict))
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Removing stopwords..."):
                df['slang_converted_no_stopwords'] = df['slang_converted'].apply(lambda x: remove_stopwords(x, custom_stopwords))
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Stemming text..."):
                df['stemmed'] = df['slang_converted_no_stopwords'].apply(stem_text)
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)
        
            with st.spinner("Finalizing processed text..."):
                df['processed_text'] = df['stemmed'].apply(lambda x: ' '.join(x))
                df['processed_text_for_sentiment'] = df['slang_converted_no_stopwords'].apply(lambda x: ' '.join(x))
                
                df = df[df['processed_text'].str.strip().astype(bool)].copy()
                if len(df) < original_row_count:
                    st.warning(f"Original rows: {original_row_count}, Removed: {original_row_count - len(df)}, Remaining data after preprocessing: {len(df)}")
                if df.empty:
                    st.error("No valid text remains after preprocessing.")
                    st.stop()
                
                step += 1
                progress_bar.progress(step / steps)
                progress_placeholder.empty()
                progress_placeholder.progress(step / steps)

                progress_placeholder.empty()
    
            # Sentiment Analysis
            with st.spinner("Analyzing sentiment..."):
                sentiments = predict_sentiment(df['processed_text_for_sentiment'].dropna().tolist(), model, tokenizer, device)
                df['sentiment_result'] = [result['sentiment'] for result in sentiments]
                df['confidence'] = [result['confidence'] for result in sentiments]
                df = df.dropna(subset=['sentiment_result'])
    
            # Tabs for Results
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Visualizations", "📥 Download Results"])
    
            # Tab 1: Data Preview
            with tab1:
                st.markdown("<h3 style='margin-bottom: -15px; margin-top: -15px'>Dataset Preview</h3><p>Here are the first few rows of your uploaded dataset:</p>", unsafe_allow_html=True)
                st.dataframe(df[original_columns], use_container_width=True)
                st.markdown("<hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 10px;' />", unsafe_allow_html=True)

            # Tab 2: Visualizations
            with tab2:
                st.markdown("""
                        <h2 style='font-size: 24px; text-align: center; margin-bottom: 10px;
                            background: linear-gradient(to right, #CCE0AC, #F0EAAC); 
                            color: black; padding: 10px; border-radius: 5px;'>
                            🧠 Insights at a Glance 📊
                        </h2>
                    """, unsafe_allow_html=True)
            
                # Sentiment Distribution Text Box
                st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#FFDAB9; border: 1px solid #000000; padding:3px; border-radius:5px;'>Sentiment Distribution</h4>", unsafe_allow_html=True)
                st.write("")
                sentiment_counts = df['sentiment_result'].value_counts()
                col1, col2, col3 = st.columns(3)
                with col1:
                    negative_count = sentiment_counts.get('negative', 0)
                    st.markdown(
                        f"""
                        <div style="background-color:#FF8A8A; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                            <span style="font-weight: 550; font-size: 18px;">Negative ☹️</span><br>
                            <span style="font-weight: 550; font-size: 35px;">{negative_count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                with col2:
                    neutral_count = sentiment_counts.get('neutral', 0)
                    st.markdown(
                        f"""
                        <div style="background-color:#F0EAAC; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                            <span style="font-weight: 550; font-size: 18px;">Neutral 😐</span><br>
                            <span style="font-weight: 550; font-size: 35px;">{neutral_count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                with col3:
                    positive_count = sentiment_counts.get('positive', 0)
                    st.markdown(
                        f"""
                        <div style="background-color:#CCE0AC; color:black; border: 1px solid #000000; margin-bottom: 15px; padding:10px; border-radius:10px; text-align:center;">
                            <span style="font-weight: 550; font-size: 18px;">Positive ☺️</span><br>
                            <span style="font-weight: 550; font-size: 35px;">{positive_count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Bar and Pie Chart Side by Side
                st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#FFDAB9; border: 1px solid #000000; padding:3px; border-radius:5px; margin-bottom: 10px;'>Sentiment Distribution Charts</h4>", unsafe_allow_html=True)
                col_bar, col_pie = st.columns(2)
                
                # Bar Chart
                with col_bar:
                    order = ['negative', 'neutral', 'positive']
                    sentiment_counts = sentiment_counts.reindex(order).fillna(0).reset_index()
                    sentiment_counts.columns = ['sentiment', 'count']
                    custom_colors = {'negative': '#FF8A8A', 'positive': '#CCE0AC', 'neutral': '#F0EAAC'}
                    colors = [custom_colors.get(label, '#d3d3d3') for label in sentiment_counts['sentiment']]
                    fig_bar = px.bar(
                        sentiment_counts,
                        x='sentiment',
                        y='count',
                        color='sentiment',
                        color_discrete_sequence=colors,
                        text='count',
                        title="Bar Chart"
                    )
                    fig_bar.update_traces(textposition='outside', textfont_size=14)
                    fig_bar.update_layout(
                        xaxis_title="Sentiment", yaxis_title="Number of Comments",
                        title_font_size=14, title_x=0.5, showlegend=False, height=400,
                        margin=dict(t=50, b=50), plot_bgcolor='white', font=dict(size=14)
                    )
                    fig_bar.update_yaxes(showgrid=False)
                    fig_bar.update_xaxes(showline=True, linewidth=1, linecolor='black')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie Chart
                with col_pie:
                    fig_pie = go.Figure(data=[
                        go.Pie(
                            labels=sentiment_counts['sentiment'],
                            values=sentiment_counts['count'],
                            textinfo='percent+label',
                            marker=dict(colors=colors, line=dict(color='black', width=1)),
                            pull=[0.02, 0.02, 0.02],
                            rotation=45
                        )
                    ])
                    fig_pie.update_layout(
                        title="Pie Chart",
                        title_font_size=14, title_x=0.5, showlegend=True, height=400,
                        margin=dict(t=50, b=50), plot_bgcolor='white', font=dict(size=14),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Download Buttons for Charts
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([0.4,2,0.4,2])
                with col_btn2:
                    bar_buf = BytesIO()
                    fig_bar.write_image(bar_buf, format="png", scale=2)
                    bar_buf.seek(0)
                    st.download_button(
                        label="📥 Download Bar Chart (PNG)",
                        data=bar_buf,
                        file_name="sentiment_distribution_bar_chart.png",
                        mime="image/png",
                        key="download_bar_chart"
                    )
                with col_btn4:
                    pie_buf = BytesIO()
                    fig_pie.write_image(pie_buf, format="png", scale=2)
                    pie_buf.seek(0)
                    st.download_button(
                        label="📥 Download Pie Chart (PNG) ",
                        data=pie_buf,
                        file_name="sentiment_pie_chart.png",
                        mime="image/png"
                    )
            
                # Word Clouds for Formalized Text
                st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#FFDAB9; border: 1px solid #000000; padding:3px; border-radius:5px; margin-bottom: 10px'>Sentiment Word Clouds</h4>", unsafe_allow_html=True)
                positive_text = ' '.join(df[df['sentiment_result'] == 'positive']['processed_text_for_sentiment'].dropna())
                negative_text = ' '.join(df[df['sentiment_result'] == 'negative']['processed_text_for_sentiment'].dropna())
                neutral_text = ' '.join(df[df['sentiment_result'] == 'neutral']['processed_text_for_sentiment'].dropna())
                sentiments_available = {
                    'positive': bool(positive_text.strip()),
                    'negative': bool(negative_text.strip()),
                    'neutral': bool(neutral_text.strip())
                }
            
                if not any(sentiments_available.values()):
                    st.warning("No text available for positive, negative, or neutral sentiments. Word clouds cannot be displayed.")
                else:
                    fig_pos, fig_neg, fig_neutral = None, None, None
                    if sentiments_available['positive']:
                        fig_pos = generate_wordcloud(positive_text, 'Greens', 'Positive Sentiment Word Cloud')
                        st.pyplot(fig_pos)
                    if sentiments_available['negative']:
                        fig_neg = generate_wordcloud(negative_text, 'Reds', 'Negative Sentiment Word Cloud')
                        st.pyplot(fig_neg)
                    if sentiments_available['neutral']:
                        fig_neutral = generate_wordcloud(neutral_text, 'Purples', 'Neutral Sentiment Word Cloud')
                        st.pyplot(fig_neutral)
                    
                    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if sentiments_available['positive']:
                            buf_pos = get_image_buffer(fig_pos)
                            st.download_button(
                                label="📥 Download Positive Word Cloud (HD)",
                                data=buf_pos,
                                file_name="wordcloud_positive_formalized.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    with col2:
                        if sentiments_available['negative']:
                            buf_neg = get_image_buffer(fig_neg)
                            st.download_button(
                                label="📥 Download Negative Word Cloud (HD)",
                                data=buf_neg,
                                file_name="wordcloud_negative_formalized.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    with col3:
                        if sentiments_available['neutral']:
                            buf_neutral = get_image_buffer(fig_neutral)
                            st.download_button(
                                label="📥 Download Neutral Word Cloud (HD)",
                                data=buf_neutral,
                                file_name="wordcloud_neutral_formalized.png",
                                mime="image/png",
                                use_container_width=True
                            )

            # Tab 3: Download Results
            with tab3:
                st.markdown("<h3 style='margin-bottom: -15px; margin-top: -15px'>Final Dataset Preview</h3><p>Preview of the dataset with sentiment results:</p>", unsafe_allow_html=True)
                preview_columns = original_columns + ['processed_text', 'sentiment_result', 'confidence']
                preview_columns = [col for col in preview_columns if col in df.columns]
                preview_df = df[preview_columns].copy()
                
                if preview_df.empty:
                    st.warning("No data available to display.")
                else:
                    st.dataframe(preview_df, use_container_width=True)
                    csv_buf = BytesIO()
                    preview_df.to_csv(csv_buf, index=False, sep=';')
                    csv_buf.seek(0)
                    st.download_button(
                        label="📥 Download Dataset with Sentiment (CSV)",
                        data=csv_buf,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )

            st.markdown(thank_you_message, unsafe_allow_html=True)
    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.exception("Error in main application")
            st.stop()
            
if __name__ == "__main__":
    main()