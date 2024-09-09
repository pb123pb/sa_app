import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go

# Set Streamlit app layout to wide
st.set_page_config(layout="wide")

# Cache the loading of the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Multilingual model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

# Load the model and tokenizer once
tokenizer, model = load_model_and_tokenizer()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(torch.tensor(scores)).numpy()
    
    sentiment_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2],
        'compound': scores[2] - scores[0]
    }
    
    return sentiment_dict

# Function to create line chart for mean compound score
def create_line_chart(mean_compound):
    sentiment_label = "Positive" if mean_compound > 0.1 else "Neutral" if mean_compound >= -0.1 else "Negative"
    label_color = "green" if sentiment_label == "Positive" else "grey" if sentiment_label == "Neutral" else "red"
    
    fig = go.Figure(go.Scatter(
        x=[-1, mean_compound, 1],
        y=[0, 0.5, 0],
        mode='lines+markers+text',
        fill='tonexty',
        line_shape='spline',
        line=dict(color='royalblue'),
        fillcolor='rgba(173, 216, 230, 0.5)',
        text=[None, f"<b>{sentiment_label}</b>", None],
        textposition="bottom center",
        textfont=dict(color=label_color, size=16)
    ))
    fig.update_layout(
        title="Mean Sentiment",
        xaxis=dict(
            range=[-1, 1], 
            tickvals=[-1, 0, 1], 
            ticktext=["-1 (Negative)", "0 (Neutral)", "1 (Positive)"],
            showline=True,
            showticklabels=True,
            ticks="outside",
            ticklen=5,
        ),
        yaxis=dict(visible=False),
        showlegend=False,
        height=300,
        margin=dict(t=60)
    )
    return fig

# Function to create sentiment distribution chart with emojis
def create_sentiment_distribution_chart(df):
    df['Sentiment Category'] = pd.Categorical(df['Sentiment Category'], categories=['Negative', 'Neutral', 'Positive'], ordered=True)
    sentiment_counts = df['Sentiment Category'].value_counts().reindex(['Negative', 'Neutral', 'Positive'])
    
    # Emojis for each category
    negative_emoji = "😢"
    neutral_emoji = "😐"
    positive_emoji = "😊"
    
    # X-axis labels with emojis
    x_labels = [
        f"Negative<br>{negative_emoji}", 
        f"Neutral<br>{neutral_emoji}", 
        f"Positive<br>{positive_emoji}"
    ]
    
    fig = go.Figure(data=[go.Bar(
        x=x_labels,
        y=sentiment_counts.values,
        text=sentiment_counts.values,
        textposition='auto'
    )])
    fig.update_layout(
        title_text='Sentiment Distribution', 
        height=300, 
        margin=dict(t=60)
    )
    return fig

# Create three columns for layout
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("assets/logo.png", width=100)  # Adjust width as needed

with col2:
    st.markdown("<h1 style='text-align: center;'>Sentiment Analysis</h1>", unsafe_allow_html=True)

# Create tabs for Sentiment Analysis
tab1, tab2 = st.tabs(["Sentiment Analysis", "Manual Sentiment Analysis"])

# Sentiment Analysis Tab
with tab1:
    st.header("Sentiment Analysis")
    
    with st.sidebar.expander("Upload and Select Data", expanded=True):
        uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'], key="uploader")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            text_column = st.selectbox('Select the text column for analysis:', df.columns, key="text_column")

            if text_column:
                df['Sentiment'] = df[text_column].apply(lambda x: analyze_sentiment(str(x)))
                sentiment_df = pd.json_normalize(df['Sentiment'])
                df = pd.concat([df.drop(columns=['Sentiment']), sentiment_df], axis=1)
                df['Sentiment Category'] = df['compound'].apply(lambda x: 'Positive' if x > 0.1 else 'Neutral' if x >= -0.1 else 'Negative')
                
                st.success("Analysis completed! Scroll down to see the results.")

    if uploaded_file is not None and text_column:
        col1, col2 = st.columns(2)
        
        with col1:
            mean_compound = df['compound'].mean()
            st.plotly_chart(create_line_chart(mean_compound), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_sentiment_distribution_chart(df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df[[text_column, 'Sentiment Category', 'negative', 'neutral', 'positive', 'compound']], height=400)
        
        with col2:
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(' '.join(df[text_column]))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

# Manual Sentiment Analysis Tab
with tab2:
    st.header("Manual Sentiment Analysis")
    
    manual_text = st.text_area("Enter text for sentiment analysis:", key="manual_text")
    if manual_text:
        result = analyze_sentiment(manual_text)
        st.write("Sentiment Scores:", result)
