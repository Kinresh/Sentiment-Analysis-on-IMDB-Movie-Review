# -*- coding: utf-8 -*-
"""

@author: Krunal Katrodiya
"""

import pickle;
import re

trained_model = pickle.load(open("Trained Models/BigramSVMModel.pickle", 'rb'))
ngram_vectorizer = pickle.load(open("Trained Models/NgramVectorizer.pickle", 'rb'))

english_stop_words = english_stop_words = ['in', 'of', 'at', 'a', 'the']
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def clean_review(review):
    review = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in review]
    review = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in review]
    review = remove_stop_words(review)
    return review

#This function will remove english stop words from data
def remove_stop_words(reviews):
    removed_stop_words = []
    for review in reviews:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

while(1):
    print("\nEnter a movie review: ")
    entered_review = input();
    
    entered_review= entered_review.split("\n")
    
    cleanned_review = clean_review(entered_review)
    
    vectorized_review = ngram_vectorizer.transform(cleanned_review)
    
    suggestedSentiment = trained_model.predict(vectorized_review)
    
    temp=""
    if(suggestedSentiment[0]==1):
        temp="Positive"
    else:
        temp="Negative"
    
    
    print("\nsuggested sentiment of entered review is: ",temp)
    
    #print("\n suggested sentiment of entered review is: ",trained_model.predict(vectorized_review))