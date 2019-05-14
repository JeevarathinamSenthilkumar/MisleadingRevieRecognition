# -*- coding: utf-8 -*-
"""
Bustamente-Murguia, Pablo
Holguin, Alexis
Shanmukhayya Totada, Basavarajaiah
Senthilkumar, Jeevarathinam

Misleading Review Recognition

Dr. Hossain
CS5362- Data Mining
May 13, 2019

Python script that analyzes a review and detects if it is a fake or a true review.

"""

import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def normalize(reviews):
    reviews = [regex_no_space.sub("", line.lower()) for line in str(reviews)]
    reviews = [regex_space.sub(" ", line) for line in reviews]
    reviews = ''.join(reviews)   
    return reviews

def remove_stop_words(review):
    review = word_tokenize(review)
    filtered_review = []
    for word in review:
        if word not in stop_words:
            filtered_review.append(word)   
    return filtered_review

def lematization(reviews):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for w in reviews:
        lemmatized_text.append(lemmatizer.lemmatize(w))
 
    return lemmatized_text

def final_clean(review_body, category):
    review_complete=[]
    
    for review in review_body:
        review_clean = normalize(review)
        review_clean_stop = remove_stop_words(review_clean)
        review_clean_lemmatized = lematization(review_clean_stop)
        review_complete.extend(review_clean_lemmatized) 
        
    return review_complete

def count(reviews, word_number):
    word_counter = Counter(reviews) 
    most_occur = word_counter.most_common(word_number) 
  
    return most_occur 

def split_verified(dataset): 
    verified_reviews = []
    non_verified_reviews = []
    for index, review in dataset.iterrows():
        if review['verified_purchase'] == 'Y':
            verified_reviews.append(review['review_body'])
        else:
            non_verified_reviews.append(review['review_body'])
    return verified_reviews, non_verified_reviews
        

def list_difference(li1, li2): 
    return (list(set(li1) - set(li2)))



file_name = 'C:\\Users\\Alexis\\Desktop\\Data Mining\\amazon_reviews_us_Shoes_v1_00.tsv'

stop_words = stopwords.words('english')
regex_no_space = re.compile("[><#$&.:;!'?\,\"()\[\]]")
regex_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    
cf = pd.read_csv(file_name, error_bad_lines = False, sep='\t', nrows=2000000)

x_train = cf

verified, non_verified = split_verified(x_train)

cleaned_review_verified = final_clean(verified, 'review_body')
cleaned_review_non_verified = final_clean(non_verified, 'review_body')

most_frequent_words_verified = count(cleaned_review_verified, 20000)
most_frequent_words_non_verified = count(cleaned_review_non_verified, 20000)

most_frequent_only_words_verified = [i[0] for i in most_frequent_words_verified]
most_frequent_only_words_non_verified = [i[0] for i in most_frequent_words_non_verified]

fake_words = (list_difference(most_frequent_only_words_non_verified, most_frequent_only_words_verified))
truth_words =   (list_difference(most_frequent_only_words_verified, most_frequent_only_words_non_verified))

fake_words_all = (list_difference(cleaned_review_non_verified, cleaned_review_verified))
truth_words_all = (list_difference(cleaned_review_verified,cleaned_review_non_verified))

list_of_no=[]
list_of_yes =[]

for count in fake_words_all:
    list_of_no.append(1)
    
for count in cleaned_review_verified:
    list_of_yes.append(0)
    
zipped_fake = list(zip(list_of_no, fake_words_all))
zipped_truth = list(zip(list_of_yes, cleaned_review_verified))
zipped_truth = zipped_truth[:16000]

mergedlist = zipped_fake + zipped_truth


df = pd.DataFrame(mergedlist, columns =['label', 'message'])

vect = CountVectorizer()  
counts = vect.fit_transform(df['message'])  

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)  


X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=53)  

model = MultinomialNB()
model.fit(X_train, y_train)  

predicted = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted))  


print("\nModel Accuracy:")
print(model.score(X_test, y_test)*100.0)


input_review = input("Please input a review: \n")
input_review = input_review.split()
input_review = final_clean(input_review, 'review_body')

input_review = np.array(input_review)

trans_review = vect.transform(input_review.ravel())

trans_review = transformer.transform(trans_review)

prediction = model.predict(trans_review)

print("finding the Probabilities: \n)
prediction = list(prediction)

fake_score = (prediction.count(1) / len(prediction)) * 100.0

true_score = (prediction.count(0) / len(prediction)) * 100.0


print("The Probability of being a Fake Review %:")
print(fake_score)

print("\nThe Probability of being a True Review %:")
print(true_score)
