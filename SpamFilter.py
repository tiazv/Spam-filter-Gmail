import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

#import data from csv
data = pd.read_csv("spam.csv")
data = data.sample(frac=1).reset_index(drop=True)

data["spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0) #dodaj novi stolpec s števkami

cv = CountVectorizer()

pod = data.Message #content
rez = data.spam #spam or ham
pod_train, pod_test, rez_train, rez_test = train_test_split(pod, rez, test_size=.01)

pod = cv.fit(pod)

#find word count and store data as a matrix
pod_train_count = cv.fit_transform(pod_train.values)

#PRINT: how many times does a word show up
#print(pod_train_count.toarray())

#train model
model = MultinomialNB()
model.fit(pod_train_count, rez_train)

pickle.dump(model, open("spam.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
clf = pickle.load(open("spam.pkl", "rb"))

#test message for prediction
msg = ["urgent click here to get 1000€"]
msg_count = cv.transform(msg)
result = model.predict(msg_count)

print("Message:")
print(msg)
print("Result of prediction:")
print(result)
print("----------------------------")

#test model
pod_test_count = cv.transform(pod_test)
accuracy1 = model.score(pod_test_count, rez_test)
accuracy2 = str(model.score(pod_test_count, rez_test) * 100)
accuracy2_str = "{:.2f}".format(float(accuracy2))

print("Accuracy of filter: " + accuracy2_str +  " %")
#print(model.score(msg_count, result))

#PROCENTI
result_probs = model.predict_proba(msg_count)[0]
spam_percent = result_probs[1] * 100
ham_percent = result_probs[0] * 100
print(f"Spam percentage: {spam_percent:.2f} %")
print(f"Ham percentage: {ham_percent:.2f} %")
