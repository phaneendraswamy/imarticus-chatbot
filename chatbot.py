import random
import json
import pickle
import numpy as np
import nltk
import csv
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json, user_name):
    if not intents_list:
        return f"Sorry {user_name}, I didn’t understand that. Try asking about our Data Science program or fees!"
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            response = random.choice(i["responses"])
            return f"{user_name}, {response}"
    return f"Oops {user_name}, something went wrong. Please try again!"


def save_user_info(name, mobile):
    file_exists = False
    try:
        with open("user_contacts.csv", "r") as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open("user_contacts.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Mobile"])
        writer.writerow([name, mobile])


# Global variable to store user info (in a real app, use a database or session)
user_info = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_chat():
    name = request.form.get("name")
    mobile = request.form.get("mobile")

    if not name or not mobile:
        return jsonify({"response": "Please provide both name and mobile number!"})
    if not mobile.isdigit() or len(mobile) < 10:
        return jsonify({"response": "Please enter a valid mobile number (at least 10 digits)!"})

    user_info["name"] = name
    user_info["mobile"] = mobile
    save_user_info(name, mobile)

    return jsonify({"response": f"Thanks {name}! Your info is saved. How can I assist you today?"})


@app.route("/chat", methods=["POST"])
def chat():
    if "name" not in user_info:
        return jsonify({"response": "Please provide your name and mobile number first!"})

    message = request.form.get("message")
    name = user_info["name"]

    if message.lower() in ["exit", "quit", "bye"]:
        response = f"Goodbye {name}! Check out imarticus.org for more details."
        user_info.clear()  # Reset for next user
        return jsonify({"response": response, "end": True})

    if not message.strip():
        return jsonify({"response": f"Please type something, {name}, so I can assist you!"})

    ints = predict_class(message)
    response = get_response(ints, intents, name)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)