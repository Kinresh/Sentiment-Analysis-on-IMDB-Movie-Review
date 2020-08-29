# Sentiment-Analysis-on-IMDB-Movie-Review

## 1. Data Overview
For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb. The data was compiled by Andrew Maas and can be found here: [IMDb Reviews](http://ai.stanford.edu/~amaas/data/sentiment/).
The data is split evenly with 25k reviews intended for training and 25k for testing your classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.
IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with ≤ 4 stars as negative and anything with ≥ 7 stars as positive. Reviews with 5 or 6 stars were left out.

## 2. Unzipping and Merging
For this part we'll be unzipping the data file dpwnloaded from [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/). To open the tar file we'll use ```my_tar = tarfile.open('aclImdb_v1.tar.gz')```. After opeing the file we'll use extrctall() i.e. ```my_tar.extractall('./RAW_Data')```. After doing all this we'll do merging all the files i.e we'll get full_train.txt file which contain all the positive and negative data togther. All these process is done in ```prepareData.py``` file.

## 3. Data cleaning and Preprocessing
Data cleaning process will remove blank/white spaces and special characters. Preprocessing part does removing all the stop words, stemming words, and converting all the data into lower case. For removing the stop words we'll use ```english_stop_words``` list available in nltk library that contains all the list of stopwords. Next step in text preprocessing is to normalize the words in your corpus by trying to convert all of the different forms of a given word into one. Two methods that exist for this are Stemming and Lemmatization.

## 4. Various Training Models
- Unigram logistic regression model
- Bigram logistic regression model
- Trigram logistic regression model
 In Unigram logistic regression model we'll get ``` Accuracy for C=0.01 is: 0.86928, Accuracy for C=0.05 is: 0.88024, Accuracy for C=0.25 is: 0.87920, Accuracy for C=0.5 is: 0.87504, and Accuracy for C=1 is: 0.8712```. (Can be found in TrainUnigramModel.py)
In Bigram logistic regression model we'll get ```Accuracy for C=0.01 is: 0.8848, Accuracy for C=0.05 is: 0.89088, Accuracy for C=0.25 is: 0.89408, Accuracy for C=0.5 is: 0.89376, and Accuracy for C=1 is: 0.89504```. (Can be found in TrainBigramLogisticModel.py)
In Trigram logistic regression model we'll get ```Accuracy for C=0.01 is: 0.8768, Accuracy for C=0.05 is: 0.88288, Accuracy for C=0.25 is: 0.884, Accuracy for C=0.5 is: 0.88416
, and Accuracy for C=1 is: 0.88352 ```. (Can be found in TrainTrigramLogisticModel.py)

Out of these three logistic regression models Bigram has most accuracy so we'll use this model to do the further analysis. In this bigram model we have used binary vectorization using binary vectory. To improve the accuracy we'll try Word count vectoriation which will give ```Accuracy for C=0.01 is: 0.8848, Accuracy for C=0.05 is: 0.88928, Accuracy for C=0.25 is: 0.87992, Accuracy for C=0.5 is: 0.87992, and Accuracy for C=1 is: 0.8896.``` (Can be found in TrainBigramWordCountModel.py) and TFIDF vectorization which will give ```Accuracy for C=0.01 is: 0.81552, Accuracy for C=0.05 is: 0.83504, Accuracy for C=0.25 is: 0.86096, Accuracy for C=0.5 is: 0.87328, and Accuracy for C=1 is: 0.87888.```. Out of these mehods word count gives more accuracy so we'll move further with the method to do the analysis.(Can be found in TrainBigramTDIDFModel.py)

So far we’ve chosen to represent each review as a very sparse vector (lots of zeros!) with a slot for every unique n-gram in the corpus (minus n-grams that appear too often or not often enough).  Linear classifiers tend to work well on very sparse datasets (like the one we have). Another algorithm that can produce great results with a quick training time are Support Vector Machines with a linear kernel. With this algorithm we'll get an ```Accuracy for C=0.001 is: 0.88112, Accuracy for C=0.005 is: 0.88672, Accuracy for C=0.01 is: 0.88624, Accuracy for C=0.05 is: 0.88016, and Accuracy for C=0.1 is: 0.87904.```(Can be found in TrainBigramSVMModel.py)

All these trained models can be found in ```Trained Model``` folder.
