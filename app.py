import streamlit as st
from transformers import pipeline
from newspaper import Article
import nltk
import time

# 1. Essential background setup for news extraction
@st.cache_resource
def setup_tools():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

setup_tools()

# 2. Page Configuration
st.set_page_config(page_title="AI News Summarizer", page_icon="📰")

# 3. Load the AI Model (Using BART for abstractive summarization)
@st.cache_resource
def load_model():
    # Use 'facebook/bart-large-cnn' for high quality
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

st.title("🤖 AI News Summarizer")
st.write("Turn long news articles into quick, readable summaries.")

# 4. User Input
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.bbc.com/news/...")

if st.button("Generate Summary"):
    if url:
        try:
            with st.spinner('Reading article...'):
                start_time = time.time()
                
                # Fetch and extract text
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Text extraction failed. Some sites block automated tools.")
                else:
                    # Summarize (Limited to 2000 chars to avoid memory issues)
                    summary = summarizer(article.text[:2000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    # Performance Metrics
                    duration = round(time.time() - start_time, 2)
                    st.subheader(f"📄 {article.title}")
                    st.success(summary_text)
                    
                    st.divider()
                    st.info(f"**Performance:** Processed in {duration}s | Original: {len(article.text.split())} words")
                    
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a URL first.")
