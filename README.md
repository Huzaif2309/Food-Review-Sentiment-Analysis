# Amazon Food Review Sentiment Analysis

This project performs sentiment analysis on Amazon food reviews using two distinct approaches: **VADER** (Valence Aware Dictionary and sEntiment Reasoner) and **RoBERTa** (Robustly Optimized BERT Pretraining Approach). The aim is to classify customer reviews as positive, negative, or neutral.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [VADER](#vader)
  - [RoBERTa](#roberta)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Customer feedback and reviews are valuable sources of information for businesses. This project focuses on analyzing sentiments in Amazon food reviews to help better understand customer opinions. Two different sentiment analysis approaches are used:

1. **VADER**: A lexicon-based rule-driven approach.
2. **RoBERTa**: A transformer-based deep learning model, fine-tuned for sentiment classification.

## Dataset

The dataset used is the **Amazon Food Reviews** dataset, containing reviews, star ratings, and other metadata related to food products sold on Amazon.

- **Source**: [Amazon Product Review Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Columns of Interest**:
  - `Text`: The actual review text.
  - `Score`: The rating given by the customer (used for supervised fine-tuning).

## Methodology

### VADER

**VADER** is a pre-built sentiment analysis tool that works well for social media texts and other similar short pieces of text. It uses a lexicon-based approach to determine sentiment polarity (positive, negative, neutral).

- **Advantages**: 
  - Fast, lightweight.
  - Great for short texts.
- **Disadvantages**:
  - Limited by the vocabulary and rules it uses.

### RoBERTa

**RoBERTa** is a deep learning-based model derived from BERT. It is trained on a massive corpus and fine-tuned for downstream tasks such as sentiment analysis.

- **Advantages**:
  - State-of-the-art performance for various NLP tasks.
  - Capable of handling nuances in longer and more complex reviews.
- **Disadvantages**:
  - Computationally expensive.
  - Requires labeled training data for fine-tuning.

## Dependencies

Ensure the following dependencies are installed before running the project:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- transformers
- torch
- vaderSentiment
- jupyter (for notebooks)

Install dependencies with:

```bash
pip install -r requirements.txt
## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Huzaif2309/Food-Review-Sentiment-Analysis.git
   cd Food-Review-Sentiment-Analysis
## Setup

1. **Download the dataset**:  
   Download the Amazon Food Reviews dataset from Kaggle and place it in the `data/` folder.

2. **Preprocess the dataset**:  
   Preprocessing involves cleaning the text, handling missing values, and splitting the data for training and testing.

3. **Run the VADER analysis**:  
   The VADER analysis can be done using the provided Jupyter notebook:
   ```bash
   jupyter notebook vader_sentiment_analysis.ipynb
## Results

| Model   | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| VADER   | 0.70     | 0.72      | 0.68   | 0.70     |
| RoBERTa | 0.85     | 0.86      | 0.83   | 0.84     |

- **VADER**: Provides reasonable performance on short reviews but struggles with complex sentences.
- **RoBERTa**: Outperforms VADER, especially for longer reviews with nuanced sentiment.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

