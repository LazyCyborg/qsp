import torch
import os
import logging
from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd


class ModelAdaptor:
    """
    A lightweight adaptor class that provides a consistent interface
    for model loading and inference, with proper error handling
    and graceful fallbacks for MPS-related issues.

    Zero-shot classification has been removed.
    """

    def __init__(
            self,
            force_cpu: bool = False,
            batch_size: int = 16,
    ):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.batch_size = batch_size

        # Determine device strategy
        self.use_mps = False
        self.use_cuda = False

        if force_cpu:
            self.device = "cpu"
            self.logger.info("Forcing CPU usage as requested")
        else:
            # Try different device options with fallbacks
            self._setup_device()

        # Initialize models lazily (only when needed)
        self._sentiment_classifier = None
        self._sentence_model = None
        self._vader_analyzer = None

        self.logger.info(f"ModelAdaptor initialized with device: {self.device}")

    def _setup_device(self):
        """Set up the compute device with proper fallbacks"""
        # First try CUDA
        if torch.cuda.is_available():
            try:
                # Test CUDA availability with a small tensor operation
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1

                self.device = "cuda"
                self.use_cuda = True
                self.logger.info("Using CUDA for computation")
                return
            except Exception as e:
                self.logger.warning(f"CUDA available but test failed: {e}")

        # Then try MPS (for Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Set environment variables that might help with MPS stability
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

                # Test MPS with a small tensor operation
                test_tensor = torch.zeros(1).to("mps")
                _ = test_tensor + 1

                self.device = "mps"
                self.use_mps = True
                self.logger.info("Using MPS for computation")
                return
            except Exception as e:
                self.logger.warning(f"MPS available but test failed: {e}")
                # Clean up environment after MPS failure
                if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
                    del os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]

        # Fallback to CPU
        self.device = "cpu"
        self.logger.info("Using CPU for computation (no GPU available or enabled)")

    def _load_sentiment_classifier(self):
        """Load the sentiment classifier with proper error handling"""
        if self._sentiment_classifier is not None:
            return self._sentiment_classifier

        self.logger.info("Loading sentiment classifier...")
        try:
            from transformers import pipeline

            # Determine device_id based on our device selection
            device_id = -1  # CPU
            if self.use_cuda:
                device_id = 0  # First CUDA device

            # Try to load with the selected device
            self._sentiment_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device_id
            )

            # Test with a simple input
            _ = self._sentiment_classifier("Test sentence")
            self.logger.info("Sentiment classifier loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading sentiment classifier: {e}")
            self.logger.info("Falling back to CPU for sentiment classifier")

            # Fallback to CPU
            try:
                self._sentiment_classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # Force CPU
                )
            except Exception as e2:
                self.logger.error(f"Failed to load sentiment classifier even on CPU: {e2}")
                # Create a dummy classifier as last resort
                #self._sentiment_classifier = self._create_dummy_sentiment_classifier()

        return self._sentiment_classifier
    '''
   def _create_dummy_sentiment_classifier(self):
        """Create a dummy sentiment classifier as a last fallback"""

        def dummy_classifier(text):
            import random
            labels = ["positive", "neutral", "negative"]
            scores = [0.6, 0.3, 0.1]
            choice = random.choices(labels, scores)[0]
            return [{"label": choice, "score": random.uniform(0.6, 0.9)}]

        self.logger.warning("Using DUMMY sentiment classifier - results will be random!")
        return dummy_classifier'''

    def _load_sentence_model(self):
        """Load the sentence transformer model with proper error handling"""
        if self._sentence_model is not None:
            return self._sentence_model

        self.logger.info("Loading sentence transformer...")
        try:
            from sentence_transformers import SentenceTransformer

            # Load model
            self._sentence_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )

            # Move to device if not CPU
            if self.device != "cpu":
                self._sentence_model = self._sentence_model.to(self.device)

            # Test with a simple input
            _ = self._sentence_model.encode(["Test sentence"])
            self.logger.info(f"Sentence transformer loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading sentence transformer: {e}")
            self.logger.info("Falling back to CPU for sentence transformer")

            # Fallback to CPU
            try:
                self._sentence_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
            except Exception as e2:
                self.logger.error(f"Failed to load sentence transformer even on CPU: {e2}")
                # Create a dummy transformer as last resort
                #self._sentence_model = self._create_dummy_sentence_transformer()

        return self._sentence_model
    '''
    def _create_dummy_sentence_transformer(self):
        """Create a dummy sentence transformer as a last fallback"""

        class DummySentenceTransformer:
            def encode(self, sentences, **kwargs):
                # Create random 384-dimensional vectors (typical dimension for small models)
                import numpy as np
                vectors = np.random.normal(0, 1, (len(sentences), 384))
                # Normalize to unit length
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / norms

            def to(self, device):
                # Just return self for device compatibility
                return self

        self.logger.warning("Using DUMMY sentence transformer - similarity will be random!")
        return DummySentenceTransformer()
    '''

    def _load_vader_analyzer(self):
        """Load the VADER sentiment analyzer"""
        if self._vader_analyzer is not None:
            return self._vader_analyzer

        self.logger.info("Loading VADER sentiment analyzer...")
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER analyzer loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading VADER analyzer: {e}")
            # Create a dummy analyzer as fallback
            #self._vader_analyzer = self._create_dummy_vader_analyzer()

        return self._vader_analyzer
    '''
    def _create_dummy_vader_analyzer(self):
        """Create a dummy VADER analyzer as a fallback"""

        class DummyVADER:
            def polarity_scores(self, text):
                import random
                pos = random.uniform(0.3, 0.7)
                neg = random.uniform(0, 0.3)
                neu = 1.0 - pos - neg
                comp = pos - neg
                return {"pos": pos, "neg": neg, "neu": neu, "compound": comp}

        self.logger.warning("Using DUMMY VADER analyzer - results will be random!")
        return DummyVADER()
        '''
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze a single text segment using all available models.
        Returns a dictionary with all analysis results.
        """
        # Skip empty text
        if not text or not text.strip():
            return self._empty_analysis_result()

        result = {"text": text, "text_length": len(text)}

        # Get RoBERTa sentiment
        try:
            sentiment_classifier = self._load_sentiment_classifier()
            sentiment_result = sentiment_classifier(text[:1000])[0]  # Truncate long text
            result["sentiment"] = sentiment_result["label"]
            result["sentiment_confidence"] = sentiment_result["score"]
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            result["sentiment"] = "neutral"
            result["sentiment_confidence"] = 0.5

        # Get VADER sentiment
        try:
            vader_analyzer = self._load_vader_analyzer()
            vader_result = vader_analyzer.polarity_scores(text)
            result["vader_neg"] = vader_result["neg"]
            result["vader_neu"] = vader_result["neu"]
            result["vader_pos"] = vader_result["pos"]
            result["vader_compound"] = vader_result["compound"]
        except Exception as e:
            self.logger.error(f"Error in VADER analysis: {e}")
            result["vader_neg"] = 0.0
            result["vader_neu"] = 1.0
            result["vader_pos"] = 0.0
            result["vader_compound"] = 0.0

        return result

    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute a similarity matrix for a list of text segments.
        Returns a numpy array of shape (len(texts), len(texts))
        """
        if not texts:
            return np.array([[]])

        try:
            sentence_model = self._load_sentence_model()

            # Get embeddings
            embeddings = sentence_model.encode(texts)

            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms

            # Compute cosine similarity matrix
            similarity_matrix = np.matmul(normalized_embeddings, normalized_embeddings.T)

            return similarity_matrix

        except Exception as e:
            self.logger.error(f"Error computing similarity matrix: {e}")
            # Return identity matrix as fallback
            return np.eye(len(texts))

    def analyze_segments(self, segments: List[str]) -> tuple:
        """
        Analyze a list of text segments and return a DataFrame with all results.
        Also computes similarity matrix and segment-level statistics.
        All theme-related code has been removed.
        """
        results = []

        # Process each segment
        for i, segment in enumerate(segments):
            result = self.analyze_text(segment)
            result["segment_id"] = i
            result["segment_text"] = segment
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Remove the duplicate "text" column while keeping "segment_text"
        if "segment_text" in df.columns:
            df = df.drop(columns=["segment_text"])

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(segments)

        # Compute raw similarity scores (average similarity to all other segments)
        raw_scores = []
        for i in range(len(segments)):
            # Get all similarities except self-similarity
            similarities = np.concatenate([
                similarity_matrix[i, :i],
                similarity_matrix[i, i + 1:]
            ]) if i < len(segments) - 1 else similarity_matrix[i, :i]

            raw_scores.append(np.mean(similarities) if len(similarities) > 0 else 0)

        # Compute z-scores
        mean_sim = np.mean(raw_scores)
        std_sim = np.std(raw_scores)
        z_scores = [(s - mean_sim) / std_sim if std_sim > 0 else 0 for s in raw_scores]

        # Add to DataFrame
        df["similarity_raw"] = raw_scores
        df["similarity_z_score"] = z_scores

        return df, similarity_matrix

    def _empty_analysis_result(self) -> Dict:
        """Return an empty analysis result for empty text"""
        result = {
            "text": "",
            "text_length": 0,
            "sentiment": "neutral",
            "sentiment_confidence": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
            "vader_pos": 0.0,
            "vader_compound": 0.0,
        }

        return result