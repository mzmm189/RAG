import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data resources
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_whitespace(text):
    """Collapse any run of whitespace (spaces, tabs, newlines) into a single space."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_unsafe_characters(text):
    """
    Remove characters that are not standard printable text or medical punctuation.
    KEPT: letters, digits, whitespace, and standard medical punctuation:
    . , ; : % ( ) - / ' " + > < = &
    Control characters and stray unicode artifacts are removed.
    Clinical numbers, dosages, and units are NOT touched.
    """
    if not isinstance(text, str):
        return ""
    allowed = re.compile(r"[^A-Za-z0-9\s\.\,\;\:\%\(\)\-\/\'\"\+\>\<\=\&]")
    return allowed.sub(" ", text)


def clean_medical_text(text):
    """Full medical cleaning pipeline preserving numbers and clinical units."""
    text = remove_unsafe_characters(text)
    return normalize_whitespace(text)


def preprocess_for_tfidf(text):
    """Lab5-style preprocessing for TF-IDF: lowercase, tokenize, drop stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    # keep alphanumeric tokens (preserves numbers like '120' or '45')
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Patient is a 45-year-old male with 120/80 mmHg BP & 500mg Amoxicillin.\n\tNext steps: follow-up."
    cleaned = clean_medical_text(sample)
    tfidf_prep = preprocess_for_tfidf(cleaned)
    print("Original  :", repr(sample))
    print("Cleaned   :", repr(cleaned))
    print("TF-IDF Prep:", repr(tfidf_prep))
