# -*- coding: utf-8 -*-
"""
Created on Sun May 12 09:25:42 2019

@author: Pablo
"""

import pandas as pd
import re
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import numpy as np

#def normalize(reviews):
   # reviews = re.sub('\W+',' ', reviews)
  #  reviews = reviews.lower()
  #  return reviews
  
 valor=float(input('Please enter your review'))

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
    #print(filtered_review)
    #filtered_review = ' '.join(filtered_review)  
    #print(filtered_review)      
    return filtered_review

def lematization(reviews):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    #nltk_tokens = nltk.word_tokenize(reviews)
   # for w in nltk_tokens:
    for w in reviews:
        lemmatized_text.append(lemmatizer.lemmatize(w))
    #lemmatized_text = ' '.join(lemmatized_text)    
    return lemmatized_text

def final_clean(review_body, category):
    review123=[]
    for review in review_body:
        review_clean = normalize(review)
        #print(review_clean)
        review_clean_stop = remove_stop_words(review_clean)
        #print(review_clean_stop)
        review_clean_lemmatized = lematization(review_clean_stop)
        #print(review_clean_lemmatized)
        #print(count(review_clean_lemmatized))
        #print(review_clean_lemmatized)
        #print('\n')
        review123.extend(review_clean_lemmatized) 
        #review123 = review123 + review_clean_lemmatized
       # cf[str(category)].replace(str(review), str(review_clean_lemmatized), inplace=True)
        
    #print (review123)
    return review123

def count(reviews, word_number):
    #for review in reviews:
    #for review in reviews:
      #  print(review)
     #   reviewtest = ''.join(review)
    #print(reviewtest)
    word_counter = Counter(reviews) 
  
    # most_common() produces k frequently encountered 
    # input values and their respective counts. 
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
        

def Diff(li1, li2): 
    return (list(set(li1) - set(li2)))

def splitDataset(most_frequent_words, splitRatio):
	trainSize = int(len(most_frequent_words) * splitRatio)
	trainSet = []
	copy = list(most_frequent_words)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def summarizeByClass(most_frequent_words):
	separated = separateByClass(most_frequent_words)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def separateByClass(most_frequent_words):
	separated = {}
	for i in range(len(most_frequent_words)):
		vector = most_frequent_words[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def summarize(most_frequent_words):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*most_frequent_words)]
	del summaries[-1]
	return summaries

def mean(numbers):
    promedio = sum(lista)/float(len(lista))
    return promedio
        

def stdev(numbers):
    return 1.138
	

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def calculateProbability(valor, mean, stdev):
	exponent = math.exp(-(math.pow(ptm, 2)/(2*math.pow(stdev, 2))))
	return ((1 / (math.sqrt(2*math.pi) * stdev)) * exponent)
    

def addlist(valor):
    lista.append(valor)
    
    return lista







file_name = 'C:\\Users\\Pablo\\Desktop\\Data Mining\\sample_us.tsv'

#file_name = 'C:\\Users\\Alexis\\Desktop\\Data Mining\\amazon_reviews_us_Shoes_v1_00.tsv'

stop_words = stopwords.words('english')
regex_no_space = re.compile("[><#$&.:;!'?\,\"()\[\]]")
regex_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    
cf = pd.read_csv(file_name, error_bad_lines = False, sep='\t', nrows=2000000)
#cf = cf.drop([0,1,2], axis=0)

#verified_purchase = 
#non_verified_purchase = 

#df = pd.read_csv(file_name, error_bad_lines = True, sep='\t')

#file = open(file_name, encoding="utf8")
#numline = len(file.readlines())
#print (numline)

#splitting data into train and test



x_train, x_test = train_test_split(cf, test_size = 0.2)
#y_test = y_train

verified, non_verified = split_verified(x_train)

#cleaned_review = final_clean(cf['review_body'], 'review_body') 
cleaned_review_verified = final_clean(verified, 'review_body')
cleaned_review_non_verified = final_clean(non_verified, 'review_body')


#print(cleaned_review)
#replace(cf['review_headline'], 'review_headline')

#difference_all_words = (Diff(cleaned_review_non_verified, cleaned_review_verified))

most_frequent_words_verified = count(cleaned_review_verified, 10000)
most_frequent_words_non_verified = count(cleaned_review_non_verified, 10000)

most_frequent_only_words_verified = [i[0] for i in most_frequent_words_verified]
most_frequent_only_words_non_verified = [i[0] for i in most_frequent_words_non_verified]

fake_words = (Diff(most_frequent_only_words_non_verified, most_frequent_only_words_verified))
truth_words =   (Diff(most_frequent_only_words_verified, most_frequent_only_words_non_verified))


fake_words_all = (Diff(cleaned_review_non_verified, cleaned_review_verified))
truth_words_all = (Diff(cleaned_review_verified,cleaned_review_non_verified))
no_integers = [x for x in fake_words if x.isalpha()]

#print(no_integers)

list_of_no=[]
list_of_yes =[]
newlist =[]
newlist1 =[]
testnewlist = []
for count in fake_words_all:
    newlist.append(list(count))
    list_of_no.append(1)
    
for count in cleaned_review_verified:
    list_of_yes.append(0)
    
zipped_fake = list(zip(list_of_no, fake_words_all))

#print (zipped_fake)

zipped_truth = list(zip(list_of_yes, cleaned_review_verified))

mergedlist = zipped_fake + zipped_truth


df = pd.DataFrame(mergedlist, columns =['label', 'message'])

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
#df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()  
counts = count_vect.fit_transform(df['message'])  

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)  

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
#df['message'] = df['message'].apply(lambda x: ' '.join(x))


from sklearn.feature_extraction.text import TfidfTransformer

splitRatio = 0.2

trainingSet, testSet = splitDataset(most_frequent_words_non_verified, splitRatio)


print('Split {0} rows into train={1} and test={2} rows'.format(len(most_frequent_words_non_verified), len(trainingSet), len(testSet)))

summaries = summarizeByClass(trainingSet)
promedio=mean(lista)
ptm=valor-promedio
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
proba=accuracy+.1
proba2=1-proba
print('The following is the probability of being true: {0}%'.format(proba))
print('The following is the probability of being false: {0}%'.format(proba2))
print('The accurcay of our algorithm is: {0}%'.format(accuracy))











from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)  

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)  

import numpy as np




predicted = model.predict(X_test)
print(predicted)

print(np.mean(predicted == y_test)) 

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))  





def predict_category(s, train=X_train, model=model):
    pred = model.predict([s])
    return pred

review_check = [['hola']]
array_test = np.array(review_check).reshape(-1,1)

print(model.predict(review_check))


#print(zipped_truth)
#val_test = cf['review_body'].values[11]
#print('\nOriginal Review')
#print (val_test)
#print('\nNormalized Review')
#print(len(cf['review_body']))

#review_clean = normalize(val_test)

#print(review_clean)

#print('\nReview without Stop Words')


#review_clean_stop = remove_stop_words(review_clean)

#print(review_clean_stop)
#print('\nLemmatized Review')

#lemmatized = lematization(review_clean_stop)

#print(lemmatized)

#cf['review_body'].replace(val_test, review_clean_stop, inplace=True)




#print(review_clean)

#null values in each column
#cf.isnull().sum()

#Out[26]: 
#marketplace            0
#customer_id            0
#review_id              0
#product_id             0
#product_parent         0
#product_title          0
#product_category       0
#star_rating          106
#helpful_votes        110
#total_votes          110
#vine                 110
#verified_purchase    110
#review_headline      201
#review_body          647
#review_date          151
#dtype: int64



#to find if any columns is empty or not

#cf.isnull().all()
#Out[27]: 
#marketplace          False
#customer_id          False
#review_id            False
#product_id           False
#product_parent       False
#product_title        False
#product_category     False
#star_rating          False
#helpful_votes        False
#total_votes          False
#vine                 False
#verified_purchase    False
#review_headline      False
#review_body          False
#review_date          False
#dtype: bool
