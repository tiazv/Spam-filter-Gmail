import imaplib
import smtplib
import email
import yaml

#GET USERNAME
with open("ne_dodaj_na_github.yml") as f:
    content = f.read()
my_credentials = yaml.load(content, Loader = yaml.FullLoader)
email_address = my_credentials["user"]
password = my_credentials["pass"]

#CONNECT TO SERVER - IMAP
imap_server = "imap.gmail.com"
imap = imaplib.IMAP4_SSL(imap_server, 993)
imap.login(email_address, password)

#CONNECT TO SERVER - SMTP
smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
smtp_server.starttls()
smtp_server.login(email_address, password)



import SpamFilter as sf
from flask import Flask, redirect, url_for, request, jsonify

app = Flask(__name__)
@app.route("/result", methods = ["POST"])
def login():
    if request.method=="POST":
        res = request.form
        spam, ham = sf.predictSpam(res["message"])
        if(res["status"]=="check"):
            print(spam, ham)
            return jsonify({"p_spam": spam, "p_ham": ham})
    else:
        print("Bad Request")

if __name__ == "__main__":
    app.run(debug = True)