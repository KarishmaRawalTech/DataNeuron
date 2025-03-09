import re
import nltk
import torch
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask App
app = Flask(__name__)

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# API Endpoint for text similarity
@app.route('/similarity', methods=['POST'])
def get_similarity():
    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')

        if not text1 or not text2:
            return jsonify({"error": "Both 'text1' and 'text2' are required."}), 400

        # Preprocess the text
        text1_clean = preprocess(text1)
        text2_clean = preprocess(text2)

        # Compute embeddings
        embedding1 = model.encode([text1_clean], convert_to_tensor=True)
        embedding2 = model.encode([text2_clean], convert_to_tensor=True)

        # Compute cosine similarity
        similarity_score = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]

        return jsonify({"similarity score": round(float(similarity_score), 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
