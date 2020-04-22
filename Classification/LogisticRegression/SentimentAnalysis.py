import numpy as np
import pandas as pd
import os
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def word_count(sentence):
    word_count_dict = {}
    for word in [aword.lower() for aword in sentence.split()]:
        if word in word_count_dict.keys():
            word_count_dict[word] = word_count_dict[word] + 1
        else:
            word_count_dict[word] = 1
    return word_count_dict


def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    predicted_labels = model.predict(data)
    # Compute the number of correctly classified examples
    is_correct = predicted_labels == true_labels
    # Then compute accuracy by dividing num_correct by total number of examples
    accuracy = len(predicted_labels[is_correct]) / len(is_correct)
    return round(accuracy, 2)


def print_model_details(model):
    coeffs = model.coef_
    print('No. of parameters: {}'.format(coeffs.shape[1]))
    positive_coeff = [coeff for coeff in coeffs[0] if coeff > 0]
    print('No of +ve parameters: {}'.format(len(positive_coeff)))


print(os.getcwd())
products = pd.read_csv('./data/amazon_baby.csv')
products = products.fillna({'review': ''})  # fill in N/A's in the review column
#products = products.head(n=50000)
products['review'] = products.review.astype(str)
products["review_without_punctuation"] = products['review'].apply(remove_punctuation)
products["word_count"] = products["review_without_punctuation"].apply(word_count)
products = products[products['rating'] != 3]
print(len(products))
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
print(products.head())

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(products, test_size=0.2, random_state=42)
print(len(train_data))
print(len(test_data))
train_y = train_data['sentiment']
test_y = test_data['sentiment']

#predictions on subset
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                    'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                    'work', 'product', 'money', 'would', 'return']
vectorizer_subset = CountVectorizer(vocabulary=significant_words, token_pattern=r'\b\w+\b')
vectorizer_subset.fit(train_data['review_without_punctuation'])
train_subsetX = vectorizer_subset.transform(train_data['review_without_punctuation'])
test_subsetX = vectorizer_subset.transform(test_data['review_without_punctuation'])

subset_sent_model = LogisticRegression().fit(train_subsetX, train_y)
print_model_details(subset_sent_model)
subset_model_accuracy = get_classification_accuracy(subset_sent_model, test_subsetX, test_y)
print('Prediction accuracy on using only significant words as training data: {}'.format(subset_model_accuracy))

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_X = vectorizer.fit_transform(train_data['review_without_punctuation'])
#print(products['review_without_punctuation'].values)
print(vectorizer.get_feature_names())
test_X = vectorizer.transform(test_data['review_without_punctuation'])

sentiment_model = LogisticRegression().fit(train_X, train_y)
print_model_details(sentiment_model)

sample_test_data = test_data.iloc[10:13, :]
print(sample_test_data)
print(sample_test_data.iloc[0]['review'])
print(sample_test_data.iloc[1]['review'])
sample_test_matrix = vectorizer.transform(sample_test_data['review_without_punctuation'])
scores = sentiment_model.decision_function(sample_test_matrix)
print(scores)
predicted_sentiment = sentiment_model.predict(sample_test_matrix)
print(predicted_sentiment)
print("Class predictions according to sklearn:" )
predicted_probab = sentiment_model.predict_proba(sample_test_matrix)
print(predicted_probab[:, 1])
model_accuracy = get_classification_accuracy(sentiment_model, test_X, test_y)
print('Prediction accuracy on entire training data: {}'.format(model_accuracy))

