# -*- coding: utf-8 -*-
"""
@author: Krunal Katrodiya
"""

from sklearn.metrics import accuracy_score
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "


#function that preprocess data and remove blank space, special characters from text, convert into lower case etc.
def preprocess_reviews(reviews):
    
    reviewsWithNoSpace = remove_spaces(reviews)
    removedStopWordsReviews = remove_stop_words(reviewsWithNoSpace)
    stemmedReviews = get_stemmed_text(removedStopWordsReviews)
    lemmatizedReviews = get_lemmatized_text(stemmedReviews)
    
    
    return lemmatizedReviews

def remove_spaces(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    return reviews

#This function will remove english stop words from data
def remove_stop_words(reviews):
    removed_stop_words = []
    for review in reviews:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

#It performs stemming operation on text to normalize 
def get_stemmed_text(reviews):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in reviews]

# It performs lemmatization operation on text to more normalize it.
def get_lemmatized_text(reviews):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in reviews]

#load training data into reviews_train list
reviews_train = []
for line in open('RAW_Data/full_train.txt', 'r',encoding="utf-8"):
    reviews_train.append(line.strip())

#load testing data into reviews_test list
reviews_test = []
for line in open('RAW_Data/full_test.txt', 'r',encoding="utf-8"):
    reviews_test.append(line.strip())
    
#reviews_train[0]

#load stop words of English language in english_stop_words
english_stop_words = stopwords.words('english')


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

#by following two lines you can compare old review and new preprocessed review
#reviews_train[0]
#reviews_train_clean[0]


from sklearn.feature_extraction.text import CountVectorizer

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=english_stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
#print(X)
testData = ngram_vectorizer.transform(reviews_test_clean)

from sklearn.svm import LinearSVC

#first 12500 reviews are positive thats why for them target is set to 1 and for remaining 12500
#which are negative reviews so that I set it target vaue for them is 0.
target = [1 if i < 12500 else 0 for i in range(25000)]

#split input data for taining into X_train and X_test and split target(Expected Output) into
#Y_train and Y_test 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, target, train_size = 0.75)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    
    svm = LinearSVC(C=c)
    svm.fit(X_train, Y_train)
    print ("Accuracy for C={} is: {}".format(c, accuracy_score(Y_test, svm.predict(X_test))))
"""
# Accuracy for C=0.001 is: 0.88112
# Accuracy for C=0.005 is: 0.88672
# Accuracy for C=0.01 is: 0.88624
# Accuracy for C=0.05 is: 0.88016
# Accuracy for C=0.1 is: 0.87904
"""  

""" As we can see our models gives best output when C=0.01. So, now we will train our final model 
and store it on disk. we used testData to test our model which contains 25000 data. and we used 
target list for to compare output with actuall output.
"""
import pickle
final_model = LinearSVC(C=0.01)
final_model.fit(X, target)
print ("Final Accuracy: {}".format(accuracy_score(target, final_model.predict(testData))))
# Final Accuracy: 0.88828

# save the model to disk
pickle.dump(final_model, open("Trained Models/BigramSVMModel.pickle", 'wb'))


"""
Following line of code is used to print most 5 discriminating words for both positive and negative
"""

feature_to_coef = {
    word: coef for word, coef in zip(
        ngram_vectorizer.get_feature_names(), final_model.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
 
#('excel', 0.30280007990862573)
#('perfect', 0.23992931361504005)
#('must see', 0.21659807802331918)
#('superb', 0.20160737404605225)
#('favorit', 0.19290568228017965)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
#('worst', -0.4002560986194035)
#('aw', -0.33500300720058107)
#('wast', -0.30855990192910854)
#('bore', -0.2945924480803718)
#('poorli', -0.2624385262708666)