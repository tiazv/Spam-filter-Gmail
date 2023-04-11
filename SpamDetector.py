#streamlit run SpamDetector.py
import pickle
import streamlit as st
from win32com.client import Dispatch
import pythoncom

pythoncom.CoInitialize()

#def speak(prediction):
    #speak = Dispatch(("SAPI.SpVoice"))
    #speak.Speak(prediction)

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

def main():
    st.title("Email Spam Classification App")
    msg = st.text_input("Enter mail:")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()    
        prediction = model.predict(vect)
        result = prediction[0]
        result_probs = model.predict_proba(vect)[0]
        spam_percent = result_probs[1] * 100
        ham_percent = result_probs[0] * 100
        if result == 1:
            if spam_percent > 60:
            	st.error("This is spam mail")
            	st.error("Spam percentage: {:.2f} %".format(spam_percent))
            	st.error("Ham percentage: {:.2f} %".format(ham_percent))
            elif spam_percent > 40:
            	st.error("This is spam mail")
            	st.warning("Spam percentage: {:.2f} %".format(spam_percent))
            	st.warning("Ham percentage: {:.2f} %".format(ham_percent))
            #speak("This is spam mail")
        else:
            if ham_percent > 60:
            	st.success("This is ham mail")
            	st.success("Spam percentage: {:.2f} %".format(spam_percent))
            	st.success("Ham percentage: {:.2f} %".format(ham_percent))
            elif ham_percent > 40:
            	st.success("This is ham mail")
            	st.warning("Spam percentage: {:.2f} %".format(spam_percent))
            	st.warning("Ham percentage: {:.2f} %".format(ham_percent))
            #speak("This is ham mail")


main()