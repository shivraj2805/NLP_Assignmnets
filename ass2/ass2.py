import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Download required NLTK data
nltk.download('punkt')

# Sample Dataset
documents = [
    "I love natural language processing",
    "I love machine learning",
    "natural language processing is amazing"
]

print("Documents:")
for doc in documents:
    print("-", doc)

# 1. Bag of Words - Count Occurrence
print("\n--- Bag of Words : Count Occurrence ---")
count_vectorizer = CountVectorizer()
bow_count = count_vectorizer.fit_transform(documents)

print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("Count Matrix:")
print(bow_count.toarray())

# 2. Bag of Words - Normalized Count Occurrence
print("\n--- Bag of Words : Normalized Count Occurrence ---")
count_vectorizer_norm = CountVectorizer(norm='l1')
bow_normalized = count_vectorizer_norm.fit_transform(documents)

print("Normalized Count Matrix:")
print(bow_normalized.toarray())

# 3. TF-IDF
print("\n--- TF-IDF ---")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# 4. Word2Vec Embeddings
print("\n--- Word2Vec Embeddings ---")

# Tokenize documents
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4
)

# Print embedding for a sample word
print("Word2Vec vector for word 'language':")
print(word2vec_model.wv['language'])

# Similar words
print("\nMost similar words to 'language':")
print(word2vec_model.wv.most_similar('language'))
