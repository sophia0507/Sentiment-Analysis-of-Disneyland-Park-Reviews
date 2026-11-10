# Sentiment Analysis of Disneyland Park Reviews

Academic final project (team of 2), Rady School of Management, UC San Diego.

Do visitors of Disneyland California, Hong Kong and Paris experience the parks differently, what do they complain about, and can machine learning and deep learning classify review sentiment reliably? This project analyzes 42K TripAdvisor reviews to answer these questions and turns the findings into recommendations for park management.

## Highlights

- **Parks differ significantly in rating.** Average rating: California 4.54, Hong Kong 4.40, Paris 4.13 (one-way ANOVA, p < 0.001; California vs. Paris t-test, p < 0.001).
- **Visitor country matters.** UK visitors gave the highest share of low ratings (about 14%); visitors from Malaysia, India and the Philippines gave the lowest (about 3-4%).
- **Top complaints in negative reviews:** wait times, ride closures and maintenance, staff service, and high ticket and food prices.
- **Best model: SVM** (TF-IDF features), with 95.75% accuracy, 97.67% F1 and 0.84 AUROC. CNN and TextCNN were competitive but did not beat the best traditional model.

## Data

- Source: [Disneyland Reviews (Kaggle, Chillar 2019)](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews), reviews and ratings from TripAdvisor.
- 42,656 raw records (17,745 California, 11,547 Paris, 8,255 Hong Kong); 37,547 remain after cleaning.
- Cleaning: removed duplicates and missing values, dropped neutral 3-star reviews, lowercased and stripped punctuation, extracted year and month.
- Labels: ratings of 4-5 are positive, 1-2 are negative. Positive reviews are the large majority.
- The dataset is not included in this repository. Download it from the Kaggle link above.

## Methods

| Step | What was done |
|---|---|
| Exploratory analysis | Rating distribution by park, ANOVA and t-tests, low-rating share by visitor country, word cloud of negative reviews |
| Text preprocessing | Cleaning, stopword removal, stemming |
| Features | TF-IDF (top 5,000 unigrams and bigrams), CountVectorizer, 100-dimensional Word2Vec embeddings |
| Models | Logistic Regression, Naive Bayes, SVM, CNN, TextCNN |
| Evaluation | 80/20 train-test split; accuracy, F1, AUROC, confusion matrices |

## Results

| Model | Accuracy | F1 | AUROC |
|---|---|---|---|
| Logistic Regression | 95.43% | 97.51% | 0.80 |
| Naive Bayes | 93.50% | 96.52% | 0.68 |
| **SVM** | **95.75%** | **97.67%** | **0.84** |
| CNN | 94.71% | 97.11% | 0.79 |
| TextCNN | 94.23% | 96.83% | 0.80 |

## Business Recommendations

1. **Queue management:** digital wait-time estimates, expanded fast-pass options, better ride scheduling.
2. **Service quality:** stronger customer-service training for park staff.
3. **Pricing and value:** seasonal discounts, family packages and more affordable dining options.

## Limitations

- Positive reviews make up roughly 89% of the labeled data, so accuracy and positive-class F1 look high even for a weak model. Per-class metrics for the negative class (precision, recall, macro-F1) give a fuller picture.
- Sentiment labels come from star ratings, not from manual annotation of the text.
- Neutral (3-star) reviews were excluded.

## Future Work

Transformer-based models such as DistilBERT, adding Shanghai Disneyland, and sentiment intensity (mild complaint vs. severe dissatisfaction).

## Tools

Python, pandas, scikit-learn, Word2Vec, CNN / TextCNN, matplotlib.

## Repository Structure

```
.
├── README.md
├── DisneylandReview.ipynb      # analysis and models
├── DisneylandReviews.csv.gz    # dataset (Kaggle, compressed)
└── Final Project_ Sentiment Analysis of Disneyland Park Reviews.pdf   # full written report
```


## Authors

- Ting-Yu Wang | [[LinkedIn URL](https://www.linkedin.com/in/sophiawm)]

- Xiaoran Wan

## My Contribution

Ting-Yu Wang worked on data cleaning and EDA, feature engineering and
the traditional models, the Word2Vec / CNN / TextCNN models, and the
written report with business recommendations.

