# ml-product-reviews-project
A complete project for classifying product review sentiment

## Sentiment Analysis – Product Review Classification

This project uses **scikit-learn** and **pandas** to train a machine learning model that performs **sentiment analysis** on product reviews.  
The model predicts whether a review is *positive*, *negative*, or *neutral* based on the review text and its length.

---

##  Project Structure
```
project/
│
├── data/
│   └── product_reviews_full.csv       # dataset with product reviews
│
├── model/
│   └── sentiment_model.pkl            # saved trained model
│
├── train_model.py                     # script for training the model
├── predict_sentiment.py               # interactive prediction script
└── README.md                          # this file
```

---

##  1. Installation and Requirements

Make sure you have **Python 3.8+** and install the required dependencies:

```bash
pip install pandas scikit-learn joblib
```

---

##  2. Training the Model

Run **`train_model.py`** to:
- Load data from `data/product_reviews_full.csv`
- Clean and preprocess text
- Train a `RandomForestClassifier`
- Save the final model to `model/sentiment_model.pkl`

```bash
python train_model.py
```

If training is successful, you’ll see:
```
Model trained and saved as 'model/sentiment_model.pkl'
```

---

##  3. Testing the Model (Interactive Mode)

Once trained, run **`predict_sentiment.py`** to enter your own reviews and get predictions interactively:

```bash
python predict_sentiment.py
```

Example session:
```
Enter review title: Amazing headphones!
Enter review text: The sound quality is fantastic and the battery lasts long.
Predicted sentiment: positive
----------------------------------------
```

Type `exit` anytime to quit.

---

##  4. Model Overview

- **review_title** → vectorized with `TfidfVectorizer`  
- **review_text** → vectorized with `TfidfVectorizer`  
- **review_length** → normalized using `MinMaxScaler`  
- Final classifier: **RandomForestClassifier**

The model is stored as a complete **scikit-learn Pipeline**, which means it includes both preprocessing and classification steps — ready for direct predictions on raw input data.

---

##  5. Important Notes

- The dataset must contain the following columns:
  ```
  review_uuid, product_name, product_price, review_title, review_text, sentiment
  ```
- Missing values are automatically dropped (`dropna()`).
- Sentiment labels are normalized to lowercase (e.g., `positive`, `negative`, `neutral`).


