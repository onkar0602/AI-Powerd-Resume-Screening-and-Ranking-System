import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if you haven't
nltk.download('stopwords')

# Sample job description
job_description = """
We are looking for a Software Engineer with experience in Python, machine learning, data structures, and algorithms.
Strong problem-solving skills, communication skills, and the ability to work in a team environment are required.
"""

# Sample resumes (in a real scenario, you'd load them from files or a database)
resumes = [
    "Experienced software engineer with Python and machine learning expertise. Skilled in algorithms and problem-solving.",
    "Data scientist with a strong background in Python and machine learning. Experienced in data analysis and algorithms.",
    "Software developer with Java expertise. Experienced in application development and full-stack programming.",
    "Python developer with experience in software engineering, machine learning, and data structures.",
    "Web developer skilled in HTML, CSS, and JavaScript with experience in front-end development."
]

# Preprocessing function (removes punctuation, converts to lowercase, and removes stopwords)
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Preprocess job description and resumes
processed_job_description = preprocess(job_description)
processed_resumes = [preprocess(resume) for resume in resumes]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Combine job description and resumes to vectorize
documents = [processed_job_description] + processed_resumes

# Fit and transform the documents into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity between the job description (first document) and all resumes (the rest)
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Rank resumes based on cosine similarity (higher similarity means more relevant)
ranked_resumes = sorted(zip(cosine_similarities[0], resumes), reverse=True, key=lambda x: x[0])

# Display ranked resumes
print("Ranking of Resumes based on Relevance to Job Description:")
for rank, (score, resume) in enumerate(ranked_resumes, 1):
    print(f"Rank {rank}: Similarity Score = {score:.4f}\n{resume}\n")
