# utils.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

# Define custom negation words
NEGATION_WORDS = {'no', 'not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'nor', 'nothing', 'cannot', "n't", "ain't", "aren't", 
                  "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", 
                  "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"}

def preprocess_text(text):
    """
    Improved text preprocessing with negation handling.
    """
    # Convert to string in case of any non-string entries
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace contractions with full forms to better handle negations
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Get stopwords excluding negation words
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = [word for word in stop_words if word not in NEGATION_WORDS]
    
    # Process tokens with negation handling
    processed_tokens = []
    negation_active = False
    
    for i, token in enumerate(tokens):
        # Check if token is a negation word
        if token in NEGATION_WORDS:
            negation_active = True
            processed_tokens.append(token)
        # Check if token is punctuation that ends negation scope
        elif token in ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}']:
            negation_active = False
            # Skip punctuation in final result
        # Add NEG_ prefix to words in negation scope
        elif negation_active and token not in filtered_stopwords:
            processed_tokens.append("NEG_" + token)
        # Keep normal words that aren't stopwords
        elif token not in filtered_stopwords:
            processed_tokens.append(token)
    
    # Remove any special characters except underscores (for NEG_ tags)
    processed_tokens = [re.sub(r'[^a-zA-Z_]', '', token) for token in processed_tokens]
    
    # Remove empty strings
    processed_tokens = [token for token in processed_tokens if token]
    
    # Join tokens back into text
    return ' '.join(processed_tokens)