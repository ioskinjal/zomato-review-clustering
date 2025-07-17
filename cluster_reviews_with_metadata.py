import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 1: Load and Merge Datasets
# -------------------------------
reviews_df = pd.read_csv("Zomato Restaurant reviews.csv")
metadata_df = pd.read_csv("Zomato Restaurant names and Metadata.csv")

# Merge on restaurant name
merged_df = pd.merge(reviews_df, metadata_df, left_on='Restaurant', right_on='Name', how='left')

# -------------------------------
# Step 2: Preprocess Review Text
# -------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

merged_df['CleanedReview'] = merged_df['Review'].apply(preprocess_text)

# -------------------------------
# Step 3: Vectorize Reviews with TF-IDF
# -------------------------------
sampled = merged_df.head(2000).copy()  # Limit for performance
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(sampled['CleanedReview'])

# -------------------------------
# Step 4: Apply KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
sampled['Cluster'] = kmeans.fit_predict(X)

# -------------------------------
# Step 5: Visualize Clusters by Restaurant
# -------------------------------
plt.figure(figsize=(12, 6))
sns.countplot(
    data=sampled,
    x='Restaurant',
    hue='Cluster',
    order=sampled['Restaurant'].value_counts().index[:15],
    palette='tab10'
)
plt.title("Top 15 Restaurants – Review Clusters")
plt.xlabel("Restaurant")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Print Sample Reviews from Each Cluster
# -------------------------------
print("\n--- Sample Review from Each Cluster ---")
for i in range(5):
    sample = sampled[sampled['Cluster'] == i]['Review'].iloc[0]
    rest = sampled[sampled['Cluster'] == i]['Restaurant'].iloc[0]
    print(f"\n🧩 Cluster {i} — {rest}:\n{sample}")