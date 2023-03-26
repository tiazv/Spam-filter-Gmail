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

pod = spam_df.Message #content
rez = spam_df.spam #spam or ham

#create train/test split
pod_train, pod_test, rez_train, rez_test = train_test_split(pod, rez, test_size=.01)

#PRINT: inspect data
print("pod_train")
print(pod_train)
print(pod_train.describe())
print("------------------------------------------------")

#find word count and store data as a matrix
cv = CountVectorizer()
pod_train_count = cv.fit_transform(pod_train.values)

#PRINT: how many times does a word show up
print("pod_train_count")
print(pod_train_count.toarray())
print("------------------------------------------------")

#train model
model = MultinomialNB()
model.fit(pod_train_count, rez_train)

#ham example for prediction
email_ham = ["meeting reward click"]
email_ham_count = cv.transform(email_ham)
email_ham_test = model.predict(email_ham_count)

#spam example for prediction
email_spam = ["reward money click movie energy meeting"]
email_spam_count = cv.transform(email_spam)
email_spam_test = model.predict(email_spam_count)

#PRINT: spam and ham prediction
print(email_spam_count)
print("should be ham (0):")
print(model.predict(email_ham_count))
print("should be spam (1):")
print(model.predict(email_spam_count))
print("------------------------------------------------")

#test model
pod_test_count = cv.transform(pod_test)
model.score(pod_test_count, rez_test)

#PRINT: accuracy
print(model.score(pod_test_count, rez_test))
print(model.score(email_ham_count, email_ham_test))
print(model.score(email_spam_count, email_spam_test))
# Predict the class probabilities for the email
email_ham_test_probs = model.predict_proba(email_ham_count)[0]
email_spam_test_probs = model.predict_proba(email_spam_count)[0]
# Calculate the percentage of spam and ham in the predicted labels
spam_percent = email_ham_test_probs[1] * 100
ham_percent = email_ham_test_probs[0] * 100

spam_percent1 = email_spam_test_probs[1] * 100
ham_percent1 = email_spam_test_probs[0] * 100

# Print the results
print(f"Spam percentage: {spam_percent:.2f}%")
print(f"Ham percentage: {ham_percent:.2f}%")

print(f"Spam percentage1: {spam_percent1:.2f}%")
print(f"Ham percentage1: {ham_percent1:.2f}%")
