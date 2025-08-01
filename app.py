import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import joblib

# Load model and vectorizer
model = joblib.load( r"C:\Users\Tejas Pund\OneDrive\Desktop\Python\MyProjects\emotion_model.pkl" )
vectorizer = joblib.load( r"C:\Users\Tejas Pund\OneDrive\Desktop\Python\MyProjects\tfidf_vectorizer.pkl")

# Set Streamlit page config
st.set_page_config(
    page_title="YouTube Emotion Analyzer",
    layout="wide",
    page_icon="🎭"
)

# Add custom CSS for style
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        font-size:40px !important;
        color: #3E3E3E;
        font-weight: bold;
    }
    .subheader {
        font-size:20px !important;
        color: #4A4A4A;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🎭 YouTube Comment Emotion Analyzer</div>", unsafe_allow_html=True)

st.write("""
Paste your YouTube comments below and discover the emotional sentiment of your audience.
Each comment will be classified as Joy, Anger, Sadness, Fear, Surprise or Love.
""")

# Text input
comments_input = st.text_area("📝 Paste YouTube Comments (one per line):", height=250)

# Define emoji map
emoji_map = {
    "joy": "😂", "anger": "😡", "sadness": "😢",
    "fear": "😨", "surprise": "😲", "love": "❤️"
}

# Sentiment function
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# If button is clicked
if st.button("🔍 Analyze Emotions"):
    if comments_input.strip() == "":
        st.warning("Please paste at least one comment.")
    else:
        # Convert to DataFrame
        comments_list = comments_input.strip().split('\n')
        df = pd.DataFrame(comments_list, columns=["Comment"])

        # Transform and predict
        X = vectorizer.transform(df["Comment"])
        df["Emotion"] = model.predict(X)
        df["Emoji"] = df["Emotion"].map(emoji_map)
        df["Sentiment"] = df["Comment"].apply(get_sentiment)

        # Show DataFrame
        st.subheader("📋 Prediction Results:")
        st.dataframe(df, use_container_width=True)

        # Emotion Distribution Bar Chart
        st.subheader("📊 Emotion Distribution")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.countplot(data=df, x="Emotion", order=df["Emotion"].value_counts().index, palette="Set2", ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Emotion")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Pie Chart of Emotions
        st.subheader("🧁 Emotion Distribution (Pie Chart)")
        emotion_counts = df["Emotion"].value_counts()
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(emotion_counts, labels=emotion_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

        # Sentiment Bar Chart
        st.subheader("📈 Sentiment Distribution")
        fig_sent, ax_sent = plt.subplots()
        sns.countplot(data=df, x="Sentiment", palette="coolwarm", ax=ax_sent)
        st.pyplot(fig_sent)

        # Word Cloud
        st.subheader("☁️ Word Cloud from Comments")
        all_words = " ".join(df["Comment"].dropna().values)
        if all_words.strip():
            stopwords = set(STOPWORDS)
            wc = WordCloud(width=800, height=400, background_color="white", colormap="Set2", stopwords=stopwords).generate(all_words)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Not enough words to generate a word cloud.")

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name='youtube_emotions.csv', mime='text/csv')

# Footer
st.markdown("""
---
**Built by Tejas** — powered by Logistic Regression + TF-IDF + Streamlit
""")
