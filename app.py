import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from Hugging Face"""
    model_url = "https://huggingface.co/MphephuSamuel/testing_models/resolve/main/imdb_model.pkl"
    
    try:
        # Create a progress bar for downloading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Downloading model from Hugging Face...")
        
        # Download the model file
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        # Create temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
            
            tmp_file_path = tmp_file.name
        
        status_text.text("🔄 Loading model...")
        progress_bar.progress(1.0)
        
        # Load the model from the temporary file
        with open(tmp_file_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return model_data["vectorizer"], model_data["classifier"]
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading model from Hugging Face: {str(e)}")
        st.error("Please check your internet connection and the model URL.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, vectorizer, classifier):
    """Predict sentiment for given text"""
    if not text.strip():
        return None, None
    
    # Transform text to TF-IDF features
    text_tfidf = vectorizer.transform([text])
    
    # Get prediction and probability
    prediction = classifier.predict(text_tfidf)[0]
    probability = classifier.predict_proba(text_tfidf)[0]
    
    return prediction, probability

def main():
    # Header
    st.title("🎬 IMDb Movie Review Sentiment Analyzer")
    st.markdown("---")
    
    # Description
    st.markdown("""
    This app uses a **Logistic Regression** model trained on the IMDb dataset to classify movie reviews as:
    - 🟢 **Positive** (score = 1) 
    - 🔴 **Negative** (score = 0)
    
    The model was trained on 40,000 movie reviews using TF-IDF vectorization with 100,000 features.
    
    🤗 **Model hosted on Hugging Face:** [MphephuSamuel/testing_models](https://huggingface.co/MphephuSamuel/testing_models)
    """)
    
    # Load model
    vectorizer, classifier = load_model()
    
    if vectorizer is None or classifier is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter a Movie Review")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Area", "Sample Reviews"],
            horizontal=True
        )
        
        if input_method == "Text Area":
            user_input = st.text_area(
                "Write your movie review here:",
                height=150,
                placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
            )
        else:
            # Sample reviews for testing
            sample_reviews = {
                "Positive Sample 1": "This movie was absolutely incredible! The cinematography was breathtaking, the acting was superb, and the storyline kept me on the edge of my seat. I would definitely recommend this to anyone who loves great filmmaking.",
                
                "Positive Sample 2": "What an amazing film! The director really outdid themselves with this masterpiece. The character development was excellent and the emotional depth really touched my heart. A must-watch!",
                
                "Negative Sample 1": "This movie was a complete waste of time. The plot was confusing, the acting was terrible, and I found myself checking my watch every few minutes. I can't believe I sat through the entire thing.",
                
                "Negative Sample 2": "Absolutely awful. The storyline made no sense, the dialogue was cringe-worthy, and the special effects looked like they were done on a shoestring budget. One of the worst movies I've ever seen."
            }
            
            selected_sample = st.selectbox("Choose a sample review:", list(sample_reviews.keys()))
            user_input = st.text_area(
                "Sample review:",
                value=sample_reviews[selected_sample],
                height=150
            )
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if user_input and user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction, probabilities = predict_sentiment(user_input, vectorizer, classifier)
                
                if prediction is not None:
                    # Display prediction
                    if prediction == 1:
                        st.success("🟢 **POSITIVE** Review")
                        st.balloons()
                    else:
                        st.error("🔴 **NEGATIVE** Review")
                    
                    # Display confidence scores
                    st.subheader("📊 Confidence Scores")
                    
                    negative_conf = probabilities[0] * 100
                    positive_conf = probabilities[1] * 100
                    
                    # Progress bars for confidence
                    st.markdown("**Negative:**")
                    st.progress(negative_conf / 100)
                    st.text(f"{negative_conf:.1f}%")
                    
                    st.markdown("**Positive:**")
                    st.progress(positive_conf / 100)
                    st.text(f"{positive_conf:.1f}%")
                    
                    # Additional info
                    max_conf = max(negative_conf, positive_conf)
                    if max_conf < 60:
                        st.warning("⚠️ Low confidence prediction. The model is unsure about this review.")
                    elif max_conf > 80:
                        st.info("✨ High confidence prediction!")
                    
            else:
                st.warning("⚠️ Please enter a movie review to analyze.")
    
    # Footer with model info
    st.markdown("---")
    
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Details:**
        - **Algorithm:** Logistic Regression
        - **Features:** TF-IDF Vectorization (100,000 max features)
        - **Training Data:** IMDb Movie Reviews Dataset (40,000 reviews)
        - **Classes:** Binary (Positive/Negative)
        - **Max Iterations:** 5,000
        - **Model Source:** [Hugging Face - MphephuSamuel/testing_models](https://huggingface.co/MphephuSamuel/testing_models)
        
        **How it works:**
        1. Model is downloaded from Hugging Face on first run
        2. Your text is converted to numerical features using TF-IDF
        3. The logistic regression model calculates probabilities
        4. The class with higher probability becomes the prediction
        """)

if __name__ == "__main__":
    main()
