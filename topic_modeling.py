import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import freeze_support
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import nltk
from nltk.corpus import stopwords

# ====================== 2. CLEAN THE TEXT ======================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)      # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', ' ', text)   # remove short words
    return ' '.join(text.split())
def get_sentiment(rating):
    if rating >= 4: return "Positive"
    elif rating == 3: return "Neutral"
    else: return "Negative"


def main():
    nltk.download('stopwords', quiet=True)

    # ====================== 1. LOAD & INSPECT ======================
    print("Step 1: Loading 1429_1.csv ...")
    df = pd.read_csv("1429_1.csv")

    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nSample reviews.text:\n", df['reviews.text'].head(3).tolist())
    print("\nMissing values in key columns:")
    print(df[['reviews.text', 'reviews.title', 'reviews.rating', 'reviews.date']].isnull().sum())

    # ====================== 2. CLEAN THE TEXT ======================
    print("\nStep 2-3: Cleaning and Preprocessing...")
    df['clean_text'] = df['reviews.text'].apply(clean_text)

    # ====================== 4. VECTORIZE ======================
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.85,
                                 stop_words=stop_words)
    X = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()

    # ====================== 5. COHERENCE SCORES (Choose Number of Topics) ======================
    print("\nStep 5: Computing coherence scores (5-15 topics)...")
    coherence_scores = []
    topic_range = range(5, 16)
    texts = [doc.split() for doc in df['clean_text']]
    dictionary = Dictionary(texts)

    for n in topic_range:
        nmf = NMF(n_components=n, random_state=42, max_iter=500)
        nmf.fit_transform(X)
        H_temp = nmf.components_

        topics = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in H_temp]
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        coherence_scores.append(score)
        print(f"Topics: {n} -> Coherence: {score:.4f}")

    # Plot elbow
    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, coherence_scores, marker='o', linewidth=2)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Scores on 1429_1.csv (NMF)')
    plt.grid(True)
    plt.show()

    # ====================== 6. TRAIN NMF WITH 8 TOPICS ======================
    n_topics = 8
    print(f"\nStep 6: Training NMF with {n_topics} topics...")
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_

    # ====================== 7. LABEL TOPICS WITH GROUP NAMES ======================
    group_names = [
        "Fire HD 8 Tablet Praise",
        "E-reading Experience",
        "Kids Gifts Beginners",
        "Charging Cables",
        "Fire TV Streaming",
        "Price Value",
        "Performance",
        "Kindle DX Leather Cover"
    ]

    print("\n=== FINAL 8 TOPICS WITH GROUP NAMES ===")
    for i in range(n_topics):
        top_words = [feature_names[j] for j in H[i].argsort()[:-16:-1]]
        print(f"\nTopic {i+1}: {group_names[i]}")
        print("Top words:", ", ".join(top_words))

        # 3 example reviews
        top_idx = W[:, i].argsort()[-3:][::-1]
        print("Example reviews:")
        for idx in top_idx:
            print(f"   - {df['reviews.text'].iloc[idx][:200]}...")

    # Assign topic to dataframe
    df['topic_id'] = np.argmax(W, axis=1)
    df['topic_name'] = df['topic_id'].map(lambda x: group_names[x])

    # ====================== 8. STATIC SENTIMENT PER TOPIC ======================
    print("\nStep 8: Static Sentiment Distribution per Topic")
    df['sentiment'] = df['reviews.rating'].apply(get_sentiment)

    static = df.groupby(['topic_name', 'sentiment']).size().unstack(fill_value=0)
    static['Total'] = static.sum(axis=1)
    static['% Positive'] = (static.get('Positive', 0) / static['Total'] * 100).round(1)
    print(static[['Positive', 'Neutral', 'Negative', 'Total', '% Positive']])

    # ====================== 9. DYNAMIC SENTIMENT OVER TIME ======================
    print("\nStep 9: Dynamic Sentiment Over Time (% Positive per Topic per Month)")

    # Convert date
    df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
    df['month'] = df['reviews.date'].dt.to_period('M')

    dynamic = df.groupby(['topic_name', 'month', 'sentiment']).size().unstack(fill_value=0)
    dynamic['Total'] = dynamic.sum(axis=1)
    dynamic['% Positive'] = (dynamic.get('Positive', 0) / dynamic['Total'] * 100).round(1)

    # Reset index for easier viewing/plotting
    dynamic = dynamic.reset_index()

    # Example: Plot % Positive trend for each topic
    plt.figure(figsize=(14, 8))
    for topic in group_names:
        data = dynamic[dynamic['topic_name'] == topic]
        if not data.empty:
            plt.plot(data['month'].astype(str), data['% Positive'], marker='o', label=topic)

    plt.xticks(rotation=45)
    plt.xlabel('Month')
    plt.ylabel('% Positive Reviews')
    plt.title('Dynamic Sentiment Trend (% Positive) per Topic over Time - 1429_1.csv')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save everything
    df.to_csv("1429_1_with_topics_sentiment.csv", index=False)
    print("\nAll done! Results saved to '1429_1_with_topics_sentiment.csv'")
    print("You now have topic_name and sentiment columns for further analysis.")


if __name__ == '__main__':
    freeze_support()
    main()