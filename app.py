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