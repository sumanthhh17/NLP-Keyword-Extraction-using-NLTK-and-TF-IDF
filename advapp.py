import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Input text
text = input("Enter a sentence: ")

# Step 1: Tokenization
words = word_tokenize(text)
print("\nTokens:")
print(words)

# Step 2: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
print("\nAfter Stopword Removal:")
print(filtered_words)

# Step 3: Convert words back to sentence
processed_text = " ".join(filtered_words)
print("\nProcessed Text:")
print(processed_text)

# Step 4: Apply TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([processed_text])

# Step 5: Extract keywords and scores
keywords = vectorizer.get_feature_names_out()
scores = tfidf_matrix.toarray()[0]

print("\nKeywords with TF-IDF Scores:")
for word, score in zip(keywords, scores):
    print(f"{word} : {score:.3f}")

# Step 6: Top Keywords
keyword_scores = list(zip(keywords, scores))
keyword_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop Keywords:")
for word, score in keyword_scores[:5]:
    print(f"{word} : {score:.3f}")