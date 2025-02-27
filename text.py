import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources if needed
try:
    nltk.download('punkt', quiet=True)
except:
    # If download fails, we'll use a simple regex-based sentence tokenizer as fallback
    pass


def sent_tokenize(text):
    """
    Tokenize text into sentences using NLTK if available,
    otherwise use a simple regex-based approach as fallback
    """
    try:
        from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
        return nltk_sent_tokenize(text)
    except:
        # Simple fallback using regex for sentence boundaries
        import re
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)


def segment_text(text, method, segment_size):
    """
    Segment the text based on specified method and size

    Args:
        text (str): The input text to segment
        method (str): The method for segmentation ('sentences', 'words', 'paragraphs')
        segment_size (int): Number of units per segment

    Returns:
        list: List of text segments
    """
    # Clean the text first
    text = re.sub(r'\s+', ' ', text).strip()

    if method == 'sentences':
        # Segment by sentences
        sentences = sent_tokenize(text)
        segments = []

        for i in range(0, len(sentences), segment_size):
            segment = ' '.join(sentences[i:i + segment_size])
            segments.append(segment)

    elif method == 'words':
        # Segment by words
        words = text.split()
        segments = []

        for i in range(0, len(words), segment_size):
            segment = ' '.join(words[i:i + segment_size])
            segments.append(segment)

    elif method == 'paragraphs':
        # Segment by paragraphs (split by double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        segments = []

        for i in range(0, len(paragraphs), segment_size):
            segment = ' '.join(paragraphs[i:i + segment_size])
            segments.append(segment)

    # Filter out empty segments
    segments = [s for s in segments if s.strip()]

    # Ensure there's at least one segment
    if not segments:
        segments = [text]

    return segments


def roberta_sentiment_analysis(classifier, segments):
    """
    Analyze sentiment using RoBERTa model

    Args:
        classifier: The RoBERTa sentiment classifier
        segments (list): List of text segments

    Returns:
        list: List of sentiment analysis results
    """
    results = []

    for segment in segments:
        if not segment.strip():
            results.append({"label": "neutral", "score": 0.0})
            continue

        # Truncate very long segments to avoid issues
        if len(segment) > 1000:
            segment = segment[:1000]

        result = classifier(segment)[0]
        results.append(result)

    return results


def vader_sentiment_analysis(analyzer, segments):
    """
    Analyze sentiment using VADER

    Args:
        analyzer: The VADER sentiment analyzer
        segments (list): List of text segments

    Returns:
        list: List of VADER sentiment scores
    """
    results = []

    for segment in segments:
        if not segment.strip():
            results.append({"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0})
            continue

        scores = analyzer.polarity_scores(segment)
        results.append(scores)

    return results


def compute_similarity_matrix(sentence_model, segments):
    """
    Compute similarity matrix between all segments

    Args:
        sentence_model: The sentence transformer model
        segments (list): List of text segments

    Returns:
        tuple: (similarity_matrix, raw_scores, z_scores)
    """
    # Get embeddings for all segments
    embeddings = sentence_model.encode(segments)

    # Normalize embeddings to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.matmul(embeddings, embeddings.T)

    # Compute raw score for each segment (average similarity to all others)
    raw_scores = []
    for i in range(len(segments)):
        # Get all similarities except the diagonal (self-similarity)
        similarities = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i + 1:]])
        raw_scores.append(np.mean(similarities))

    # Compute z-scores of raw similarity scores
    mean_sim = np.mean(raw_scores)
    std_sim = np.std(raw_scores)

    # Handle case where std is 0
    if std_sim == 0:
        z_scores = np.zeros_like(raw_scores)
    else:
        z_scores = (raw_scores - mean_sim) / std_sim

    return similarity_matrix, raw_scores, z_scores


def create_dataframe(segments, roberta_results, vader_results,
                     similarity_raw, similarity_z):
    """
    Create a DataFrame from all analysis results.

    Args:
        segments (list): List of text segments
        roberta_results (list): RoBERTa sentiment results
        vader_results (list): VADER sentiment results
        similarity_raw (list): Raw similarity scores
        similarity_z (list): Z-scores of similarity

    Returns:
        DataFrame: Combined analysis results
    """
    import pandas as pd
    data = []

    for i, segment in enumerate(segments):
        # Create entry
        entry = {
            "segment_id": i,
            "segment_text": segment,
            "segment_length": len(segment),

            # RoBERTa sentiment
            "sentiment": roberta_results[i]["label"],
            "sentiment_confidence": roberta_results[i]["score"],

            # VADER sentiment
            "vader_neg": vader_results[i]["neg"],
            "vader_neu": vader_results[i]["neu"],
            "vader_pos": vader_results[i]["pos"],
            "vader_compound": vader_results[i]["compound"],

            # Similarity
            "similarity_raw": similarity_raw[i],
            "similarity_z_score": similarity_z[i],

            # Position
            "position": i / len(segments) if len(segments) > 1 else 0.5
        }

        data.append(entry)

    return pd.DataFrame(data)