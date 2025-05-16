import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import json
import logging
import os
from io import BytesIO
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Define base directory and important paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_PATH = os.path.join(BASE_DIR, 'Preprocessing')
STEMMER_PATH = os.path.join(PREPROCESSING_PATH, 'Stemmer', 'mpstemmer', 'mpstemmer')
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

from mpstemmer import MPStemmer

# Setup NLTK data
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure NLTK data directory exists
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

# Download punkt_tab if not present
try:
    if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers', 'punkt_tab')):
        nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH, quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK data 'punkt_tab': {str(e)}")
    st.stop()

# Verify NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.error("NLTK 'punkt_tab' not found. Ensure the download was successful and the path is correct.")
    st.stop()

# Setup logging for unmatched_slang.log
logging.basicConfig(filename='unmatched_slang.log', level=logging.INFO, filemode='w')

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_sentiment_model():
    try:
        model_name = "johannawawi/v3_balanced_dataset_fine-tuning-java-indo-sentiment-analysist-3-class"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face: {str(e)}")
        st.stop()

# Load the sentiment model and tokenizer
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

def convert_to_slang(text, slang_dict, debug=False):
    if not isinstance(text, list) or not text:
        return []
    text_str = ' '.join(str(word) for word in text if word is not None)
    SLANG_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b', re.IGNORECASE)
    text_str = SLANG_PATTERN.sub(lambda x: slang_dict[x.group().lower()], text_str)
    return [word.lower() for word in text_str.split()]

def stem_text(document):
    return [stemmer.stem(term) for term in document]

# Function to predict sentiment using the loaded model
def predict_sentiment(texts, batch_size=16):
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
            st.warning(f"Failed to analyze sentiment for batch: {str(e)}")
            logging.error(f"Sentiment analysis error in batch {i}: {str(e)}")
            for _ in batch_texts:
                results.append({"sentiment": "neutral", "confidence": 0.0})
        
        progress = min((i + len(batch_texts)) / len(texts), 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return results

# Custom CSS for improved UI
st.markdown("""
    <style>
    body { font-family: 'Inter', 'Helvetica', 'Arial', sans-serif; color: #333; }
    h1 { font-size: 32px; font-weight: 700; font-family: 'Inter', sans-serif}
    h3 { font-size: 24px; font-weight: 600; font-family: 'Inter', sans-serif}
    h4 { font-size: 16px; font-weight: 600; font-family: 'Inter', sans-serif}
    p { font-size: 18px; line-height: 1.5; font-family: 'Inter', sans-serif}
    span {font-family: 'Inter', sans-serif; font-weight: 600}
    .stTabs { 
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef); 
        padding: 12px; 
        border-radius: 5px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: sticky; 
        top: 0; 
        z-index: 100;
    }
    button {
        font-size: 13px !important;
        padding: 8px 12px !important;
        font-family: 'Inter', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# Application title
st.markdown("""
    <div style="
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin: 20px 0;
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif;
    ">
        <h1 style="font-size: 32px; font-weight: 700; color: #333; margin-bottom: 10px;">
            🧠 Social Media Sentiment Analysis App
        </h1>
        <p style="font-size: 18px; line-height: 1.5; color: #333; margin: 0 auto; max-width: 600px; text-align: justify">
            Upload your CSV or Excel file containing social media data, 
            and get sentiment insights using Natural Language Processing (NLP). 
            This app helps you understand public sentiment from text data in a simple and efficient way.
        </p>
    </div>
""", unsafe_allow_html=True)

# Divider
st.markdown(
    """
    <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader(
    "**📁 Upload Your Dataset to Start**" \
    "   \n Only .xlsx or .csv files are supported", 
    type=["xlsx", "csv"],
    help="Upload an Excel (.xlsx) or CSV (.csv) file containing your dataset. Make sure the first row includes column headers and the data is well-structured."
)

if uploaded_file is not None:
    try:
        # Validate preprocessing folder
        if not os.path.exists(PREPROCESSING_PATH):
            st.error(f"Preprocessing folder not found at {PREPROCESSING_PATH}!")
            st.stop()

        # Validate stemmer folder
        if not os.path.exists(STEMMER_PATH):
            st.error(f"Stemmer folder not found at {STEMMER_PATH}! Ensure the mpstemmer folder exists.")
            st.stop()

        # Load slang dictionary files
        slang_file1_path = os.path.join(PREPROCESSING_PATH, 'slangword.txt')
        slang_file2_path = os.path.join(PREPROCESSING_PATH, 'new_kamusalay.txt')

        if not os.path.exists(slang_file1_path):
            st.error(f"slangword.txt not found at {PREPROCESSING_PATH}!")
            st.stop()
        if not os.path.exists(slang_file2_path):
            st.error(f"new_kamusalay.txt not found at {PREPROCESSING_PATH}!")
            st.stop()

        try:
            with open(slang_file1_path, "r", encoding='utf-8') as f1:
                slang_dict1 = json.load(f1)
            with open(slang_file2_path, "r", encoding='utf-8') as f2:
                slang_dict2 = json.load(f2)
        except json.JSONDecodeError:
            st.error("Format of slangword.txt or new_kamusalay.txt is not valid JSON!")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load slang dictionary files: {str(e)}")
            st.stop()

        combined_slang_dict = {k.lower(): v for k, v in {**slang_dict1, **slang_dict2}.items()}

        # Read dataset
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        original_columns = list(df.columns)

        # Select text column
        text_column = st.selectbox(
            "**Select the text column for analysis (e.g., full_text)**",
            df.columns,
            index=0,
            help="Choose the column from your dataset that contains the main text (e.g., product reviews, tweets, or comments)."

        )
        texts = df[text_column].dropna().astype(str).tolist()
        valid_texts = [t for t in texts if t.strip() and isinstance(t, str)]

        if not valid_texts:
            st.error(f"The selected column '{text_column}' contains no valid text data. Please choose a column with text content.")
            st.stop()

        # Preprocessing 
        with st.spinner("Processing text data..."):
            progress_bar = st.progress(0)
            steps = 8
            current_step = 0

            df['cleaned_text'] = df[text_column].apply(clean_text)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['emoji_removed'] = df['cleaned_text'].apply(remove_emoji)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['repeated_chars_removed'] = df['emoji_removed'].apply(replace_repeated_chars)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['lowercased'] = df['repeated_chars_removed'].apply(lowercase_text)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['tokenized'] = df['lowercased'].apply(tokenize_text)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['slang_converted'] = df['tokenized'].apply(lambda x: convert_to_slang(x, combined_slang_dict, debug=True))
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['stemmed'] = df['slang_converted'].apply(stem_text)
            current_step += 1
            progress_bar.progress(current_step / steps)

            df['processed_text'] = df['stemmed'].apply(lambda x: ' '.join(x))
            
            # Filter out empty or whitespace-only processed_text
            original_row_count = len(df)
            df = df[df['processed_text'].str.strip().astype(bool)].copy()
            removed_rows = original_row_count - len(df)
            if removed_rows > 0:
                st.warning(f"Removed {removed_rows} rows with empty or whitespace-only text after preprocessing.")
                logging.warning(f"Removed {removed_rows} empty rows after preprocessing. Sample empty texts: {df[df['processed_text'].str.strip() == ''][text_column].head().tolist()}")
            if df.empty:
                st.error("After preprocessing, no valid text remains. Please check your dataset or preprocessing steps.")
                st.stop()

            current_step += 1
            progress_bar.progress(current_step / steps)

            progress_bar.empty()

        # Sentiment Analysis
        with st.spinner("Analyzing sentiment... This may take a moment for large datasets."):
            processed_texts = df['processed_text'].dropna().tolist()
            progress_bar = st.progress(0)
            sentiments = predict_sentiment(processed_texts)
            df['sentiment_result'] = [result['sentiment'] for result in sentiments]
            df['confidence'] = [result['confidence'] for result in sentiments]
            df = df.dropna(subset=['sentiment_result'])

        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview & Preprocessing", "📈 Visualizations", "📥 Download Results"])

        # Tab 1: Data Preview and Preprocessing Results
        with tab1:
            # Display data preview
            st.markdown("<h3 style='font-size: 21px;'>Dataset Preview</h3>", unsafe_allow_html=True)
            st.markdown("<p style='margin-top: -15px; font-size: 16px'>Here are the first few rows of your uploaded dataset:</p>", unsafe_allow_html=True)
            st.dataframe(df[original_columns], use_container_width=True)

            # Divider
            st.markdown(
                """
                <hr style='border: 1px solid #ccc; margin-top: -5px; margin-bottom: 10px;' />
                """,
                unsafe_allow_html=True
            )

            # Display preprocessing results
            st.markdown("<h3 style='font-size: 21px; margin-top: -20px'>Preprocessing Results</h3>", unsafe_allow_html=True)
            st.markdown("<p style='margin-top: -15px; font-size: 16px; margin-bottom: -15px'>Here is a preview of the preprocessed text:</p>", unsafe_allow_html=True)
            st.dataframe(df[[text_column, 'cleaned_text', 'lowercased', 'slang_converted', 'processed_text']], use_container_width=True)

        # Tab 2: Visualizations
        with tab2:
            st.markdown("<h2 style='font-size: 24px; text-align: center; margin-bottom: 10px; border: 1px solid grey; padding: 5px'>Visual Summary of Findings</h2>", unsafe_allow_html=True)

            # Sentiment Distribution Text Box (tetap sama, no Plotly here)
            st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#9EC6F3; border: 1px solid #000000; padding:3px; border-radius:5px;'>Sentiment Distribution</h4>", unsafe_allow_html=True)
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
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                neutral_count = sentiment_counts.get('neutral', 0)
                st.markdown(
                    f"""
                    <div style="background-color:#F0EAAC; color:black; border: 1px solid #000000; padding:10px; border-radius:10px; text-align:center;">
                        <span style="font-weight: 550; font-size: 18px;">Neutral 😐</span><br>
                        <span style="font-weight: 550; font-size: 35px;">{neutral_count}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                positive_count = sentiment_counts.get('positive', 0)
                st.markdown(
                    f"""
                    <div style="background-color:#CCE0AC; color:black; border: 1px solid #000000; margin-bottom: 15px; padding:10px; border-radius:10px; text-align:center;">
                        <span style="font-weight: 550; font-size: 18px;">Positive ☺️</span><br>
                        <span style="font-weight: 550; font-size: 35px;">{positive_count}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Bar Chart for Sentiment Distribution with Plotly
            st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#9EC6F3; border: 1px solid #000000; padding:3px; border-radius:5px;'>Bar Chart of Sentiment Distribution</h4>", unsafe_allow_html=True)

            order = ['negative', 'neutral', 'positive']
            sentiment_counts = sentiment_counts.reindex(order).fillna(0).reset_index()
            sentiment_counts.columns = ['sentiment', 'count']

            custom_colors = {
                'negative': '#FF8A8A',
                'positive': '#CCE0AC',
                'neutral': '#F0EAAC'
            }
            colors = [custom_colors.get(label, '#d3d3d3') for label in sentiment_counts['sentiment']]

            # Create Plotly Bar Chart
            fig_bar = px.bar(
                sentiment_counts,
                x='sentiment',
                y='count',
                color='sentiment',
                color_discrete_sequence=colors,
                text='count',
                title="Sentiment Distribution"
            )
            fig_bar.update_traces(textposition='outside', textfont_size=10)
            fig_bar.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Number of Comments",
                title_font_size=13,
                title_x=0.5,
                showlegend=False,
                height=500,
                margin=dict(t=50, b=50),
                plot_bgcolor='white',
                font=dict(size=10),
            )
            fig_bar.update_yaxes(showgrid=False)
            fig_bar.update_xaxes(showline=True, linewidth=1, linecolor='black')

            # Display Plotly figure in Streamlit
            st.plotly_chart(fig_bar, use_container_width=True)

            # Prepare download button for Bar Chart
            bar_buf = BytesIO()
            fig_bar.write_image(bar_buf, format="png", scale=3)
            bar_buf.seek(0)
            bar_col1, bar_col2, bar_col3 = st.columns([6, 1.5, 4])
            with bar_col3:
                st.download_button(
                    label="📥 Download Bar Chart (PNG)",
                    data=bar_buf,
                    file_name="sentiment_distribution_bar_chart.png",
                    mime="image/png"
                )

            # Pie Chart for Sentiment with Plotly
            st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#9EC6F3; border: 1px solid #000000; padding:3px; border-radius:5px;'>Pie Chart of Sentiment Distribution</h4>", unsafe_allow_html=True)

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
                title="Sentiment Distribution",
                title_font_size=13,
                title_x=0.5,
                showlegend=True,
                height=500,
                margin=dict(t=50, b=50),
                plot_bgcolor='white',
                font=dict(size=10)
            )

            # Display Pie Chart
            st.plotly_chart(fig_pie, use_container_width=True)

            # Prepare download button for Pie Chart
            pie_buf = BytesIO()
            fig_pie.write_image(pie_buf, format="png")
            pie_buf.seek(0)
            pie_col1, pie_col2, pie_col3 = st.columns([6, 1.5, 4])
            with pie_col3:
                st.download_button(
                    label="📥 Download Pie Chart (PNG)",
                    data=pie_buf,
                    file_name="sentiment_pie_chart.png",
                    mime="image/png"
                )

            # Word Cloud Visualization
            st.markdown("<h4 style='text-align: center; font-size: 20px; background-color:#9EC6F3; border: 1px solid #000000; padding:3px; border-radius:5px;'>Sentiment Word Clouds</h4>", unsafe_allow_html=True)

            # Function to generate word cloud
            def generate_wordcloud(text, colormap, title):
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
                fig, ax = plt.subplots(dpi=800)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(title, fontsize=12, pad=10)
                return fig

            # Function to create image buffer for download
            def get_image_download_link(fig, file_name):
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=800, bbox_inches='tight')
                buf.seek(0)
                return buf
            
            # Collect text for each sentiment
            positive_text = ' '.join(df[df['sentiment_result'] == 'positive']['processed_text'].dropna())
            negative_text = ' '.join(df[df['sentiment_result'] == 'negative']['processed_text'].dropna())
            neutral_text = ' '.join(df[df['sentiment_result'] == 'neutral']['processed_text'].dropna())

            # Check which sentiments have valid text
            sentiments_available = {
                'positive': bool(positive_text.strip()),
                'negative': bool(negative_text.strip()),
                'neutral': bool(neutral_text.strip())
            }

            # If no sentiments are available
            if not any(sentiments_available.values()):
                st.warning("No text available for positive, negative, or neutral sentiments. Word clouds cannot be displayed.")
            else:
                # Initialize variables to store figures for download buttons
                fig_pos, fig_neg, fig_neutral = None, None, None

                # Display word clouds vertically in a single column
                if sentiments_available['positive']:
                    fig_pos = generate_wordcloud(positive_text, 'Greens', 'Positive Sentiment Word Cloud')
                    st.pyplot(fig_pos)

                if sentiments_available['negative']:
                    fig_neg = generate_wordcloud(negative_text, 'Reds', 'Negative Sentiment Word Cloud')
                    st.pyplot(fig_neg)

                if sentiments_available['neutral']:
                    fig_neutral = generate_wordcloud(neutral_text, 'Purples', 'Neutral Sentiment Word Cloud')
                    st.pyplot(fig_neutral)

                # Add a three-column table for download buttons
                with st.container():
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)  # Add spacing
                    col1, col2, col3 = st.columns([1, 1, 1])

                    # Positive download button
                    with col1:
                        if sentiments_available['positive']:
                            buf_pos = get_image_download_link(fig_pos, "wordcloud_positive.png")
                            st.download_button(
                                label="📥 Download Positive Word Cloud (HD)",
                                data=buf_pos,
                                file_name="wordcloud_positive.png",
                                mime="image/png",
                                use_container_width=True
                            )

                    # Negative download button
                    with col2:
                        if sentiments_available['negative']:
                            buf_neg = get_image_download_link(fig_neg, "wordcloud_negative.png")
                            st.download_button(
                                label="📥 Download Negative Word Cloud (HD)",
                                data=buf_neg,
                                file_name="wordcloud_negative.png",
                                mime="image/png",
                                use_container_width=True
                            )

                    # Neutral download button
                    with col3:
                        if sentiments_available['neutral']:
                            buf_neutral = get_image_download_link(fig_neutral, "wordcloud_neutral.png")
                            st.download_button(
                                label="📥 Download Neutral Word Cloud (HD)",
                                data=buf_neutral,
                                file_name="wordcloud_neutral.png",
                                mime="image/png",
                                use_container_width=True
                            )

        # Tab 3: Download Results
        with tab3:
            st.markdown("<h2 style='font-size: 21px;'>Final Dataset Preview</h2>", unsafe_allow_html=True)
            st.markdown("<p style='margin-top: -15px; font-size: 16px'>Preview of the original dataset with preprocessed text, sentiment, and confidence scores:</p>", unsafe_allow_html=True)

            # Define columns to display: original input columns plus derived columns
            preview_columns = original_columns + ['processed_text', 'sentiment_result', 'confidence']
            # Ensure only existing columns are included
            preview_columns = [col for col in preview_columns if col in df.columns]
            preview_df = df[preview_columns].copy()

            # Check if the DataFrame is empty
            if preview_df.empty:
                st.warning("No data available to display in the final dataset preview. Please check your dataset or processing steps.")
            else:
                # Display all rows of the DataFrame
                st.dataframe(preview_df, use_container_width=True)

                # Prepare CSV for download
                csv_buf = BytesIO()
                preview_df.to_csv(csv_buf, index=False, sep=';')
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
        st.error(f"An error occurred: {str(e)}")
        logging.exception("Error in main processing block")
        st.stop()
