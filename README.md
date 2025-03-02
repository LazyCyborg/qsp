# QSP (Quantitative Semantic Analysis Python)

A streamlit-based application for analyzing interview texts using transformer models, sentiment analysis, and semantic similarity metrics.

## Features

- Text segmentation by sentences, words, or paragraphs
- Sentiment analysis using RoBERTa and VADER
- Semantic similarity analysis between text segments
- Statistical analysis with z-scores to identify significant patterns
- Interactive visualizations for all analyses
- Export results in CSV format

## Create a virtual environment (strongly recommended)
For more info see: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

```bash
conda create -n qsp python=3.10
conda activate qsp
```

## Clone the repo and install the packages

```bash
git clone https://github.com/LazyCyborg/qsp.git
cd qsp
pip install -r requirements.txt
```

## Activate your virtual environment (if not already activated):
- Open the terminal and re-activate your environment 
- You can see which environment you are in by looking at the prefix before your username

Example ("base" is the default conda environment of your system):

(base) username@xxx ~ % 

```bash
conda activate qsp
```

## Run the app

```bash
cd qsp
streamlit run app.py
```

The app will open in your default web browser

## Analysis Methodology

### Text Segmentation

The application segments text using three methods:
1. **Sentences**: Splits text into sentences using NLTK's sentence tokenizer
2. **Words**: Splits text by individual words
3. **Paragraphs**: Splits text by paragraph breaks

You can configure the segment size (number of units per segment) to control granularity.

### Sentiment Analysis

QSP uses two complementary sentiment analysis methods:

1. **RoBERTa Transformer Model**: 
   - Uses [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
   - Provides classification as "positive," "neutral," or "negative"
   - Includes confidence scores for classifications
   - Fine-tuned on Twitter data

2. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - Rule-based sentiment analyzer specifically tuned for social media
   - Provides compound scores from -1 (extremely negative) to +1 (extremely positive)
   - Also provides positive, negative, and neutral component scores
   - Particularly effective for short, informal texts
   - Implementation from [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

The combination of these approaches provides both deep-learning-based and lexicon-based analysis perspectives.

### Semantic Similarity Analysis

The application uses [Sentence Transformers](https://sbert.net/) to compute semantic similarity between text segments. The process works as follows:

1. **Vector Embeddings**: Each text segment is encoded into a high-dimensional vector using the multilingual MiniLM sentence transformer model
2. **Similarity Matrix**: Cosine similarity is computed between all segment pairs, creating a similarity matrix
3. **Raw Similarity Scores**: For each segment, an average similarity score to all other segments is calculated
4. **Z-Score Calculation**: These raw scores are converted to z-scores to identify statistically significant patterns

#### Z-Score Calculation Details:
```python
# Compute z-scores from raw similarity scores
mean_sim = np.mean(raw_scores)  # Mean of all raw similarity scores
std_sim = np.std(raw_scores)    # Standard deviation of raw similarity scores
z_scores = [(s - mean_sim) / std_sim if std_sim > 0 else 0 for s in raw_scores]
```

Z-scores tell you how many standard deviations a segment's similarity is from the mean. Scores beyond ±1.96 (shown in the visualizations with red dotted lines) are statistically significant at the 95% confidence level, indicating segments that are notably different from others in semantic content.

## Visualizations

QSP provides several interactive visualizations:

1. **Similarity Matrix Heatmap**: Shows the similarity between each pair of segments
2. **Sentiment Analysis Timeline**: Tracks sentiment changes across segments
3. **Segment Similarity Analysis**: Shows raw similarity scores and z-scores for identifying significant patterns

## Closing the app
To close the app, press **Ctrl + C** in the terminal and close the browser tab.

**Note that the first time one runs the app it may take some time since the transformer models need to be downloaded.**

## Output Formats
All analysis results can be exported as CSV format for further processing.

## Model Requirements

Some models require additional packages that will be automatically installed with requirements.txt:
- NLTK for text processing (punkt tokenizer)
- Transformers library for RoBERTa model
- Sentence-Transformers for semantic similarity
- VaderSentiment for rule-based sentiment analysis

## Citing

If using the app in research, please cite this GitHub repository in your references along with the following resources:

- RoBERTa Sentiment Model: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- VADER Sentiment: [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- Sentence Transformers: [SBERT](https://sbert.net/)

## License

MIT License

## Author

Alexander Engelmark, Linköping University
