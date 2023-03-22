#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#import data from csv
spam_df = pd.read_csv("spam.csv")
spam_df = spam_df.sample(frac=1).reset_index(drop=True)

#add new column
spam_df["spam"] = spam_df["Category"].apply(lambda x: 1 if x == "spam" else 0)

#PRINT: how many spam/ham, how many different, most common, how many most common
print(spam_df)
print(spam_df.groupby("Category").describe())
print("------------------------------------------------")

#create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam)

#PRINT: inspect data
print(x_train)
print(x_train.describe())
print("------------------------------------------------")

#find word count and store data as a matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

#PRINT: how many times does a word show up
print(x_train_count.toarray())
print("------------------------------------------------")

#train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

#ham example for prediction
email_ham = ["hey how are you? "]
email_ham_count = cv.transform(email_ham)
email_ham_test = model.predict(email_ham_count)

#spam example for prediction
email_spam = ["hey reward click"]
email_spam_count = cv.transform(email_spam)
email_spam_test = model.predict(email_spam_count)

#PRINT: spam and ham prediction
print(email_spam_count.toarray())
print("should be ham (0):")
print(model.predict(email_ham_count))
print("should be spam (1):")
print(model.predict(email_spam_count))
print("------------------------------------------------")

#test model
x_test_count = cv.transform(x_test)
model.score(x_test_count, y_test)

#PRINT: accuracy
#print(model.score(x_test_count, y_test))
print(model.score(email_ham_count, email_ham_test))
