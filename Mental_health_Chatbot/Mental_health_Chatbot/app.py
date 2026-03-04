import random
import json
import pickle
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
from keras.models import load_model

from flask import render_template, request, redirect, url_for, session, jsonify

from database_setup import app, db, User, Chat, Message

# Ensure static folder is set
app.static_folder = 'static'

# Secret key for sessions (if you didn't set it in database_setup.py)
# You can change this string to any random value
app.config['SECRET_KEY'] = 'change_this_to_a_random_secret_key'

# --------- Crisis keywords (simple safety layer) --------- #
CRISIS_KEYWORDS = [
    "kill myself",
    "end my life",
    "suicide",
    "hurt myself",
    "hurt someone",
    "kill him",
    "kill her",
    "kill them",
    "i want to die",
    "i don't want to live anymore",
    "i do not want to live anymore"
]

CRISIS_MESSAGE = (
    "I'm really glad you reached out and I'm so sorry you're feeling this way.\n\n"
    "I'm not a human and I can't handle emergencies, but your safety is very important.\n"
    "Please contact your local emergency services or a crisis hotline, or reach out to a "
    "trusted person near you right now."
)

# --------- Load trained model and data --------- #
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


# -------------------- Chatbot ML Logic -------------------- #

def clean_up_sentence(sentence: str):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence: str, words_list, show_details: bool = False):
    """Return bag-of-words vector for the sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence: str, model):
    """Predict intent class for a given sentence using the trained model."""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    """Pick a random canned response based on the predicted intent."""
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that."


def contains_crisis_phrase(text: str) -> bool:
    """Return True only if a crisis keyword matches as a whole word/phrase."""
    text = text.lower()
    for kw in CRISIS_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def chatbot_response(msg: str) -> str:
    """Main chatbot logic — ML + rule-based crisis handling."""
    print(f"chatbot_response received: {msg}")

    # ---- FIXED CRISIS DETECTION ----
    if contains_crisis_phrase(msg):
        print("Crisis phrase detected")
        return CRISIS_MESSAGE

    # ---- ML intent classification ----
    ints = predict_class(msg, model)
    print("Predicted intents:", ints)
    res = getResponse(ints, intents)
    print("Chatbot response:", res)

    return res


# -------------------- Helper: get current user -------------------- #

def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.get(user_id)


def get_or_create_active_chat(user: User) -> Chat:
    """Get the current active chat for this user, or create one."""
    chat_id = session.get("active_chat_id")
    chat = None

    if chat_id:
        chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()

    # If no active chat yet, create a new one
    if not chat:
        chat = Chat(user_id=user.id, title=None)
        db.session.add(chat)
        db.session.commit()
        session["active_chat_id"] = chat.id

    return chat


# -------------------- Auth Routes (Register / Login / Logout) -------------------- #

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            return "All fields are required.", 400

        if password != confirm:
            return "Passwords do not match.", 400

        existing = User.query.filter_by(email=email).first()
        if existing:
            return "This email is already registered. Please log in.", 400

        user = User(name=name, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session.pop("active_chat_id", None)

        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return "Invalid email or password.", 400

        session["user_id"] = user.id
        session.pop("active_chat_id", None)

        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("active_chat_id", None)
    return redirect(url_for("login"))


# ---- Profile ---- #
@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    message = None
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        new_password = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        # Basic validation
        if not name:
            error = "Name cannot be empty."
        elif new_password or confirm:
            if new_password != confirm:
                error = "New passwords do not match."
            elif len(new_password) < 6:
                error = "New password should be at least 6 characters."

        if not error:
            # Update name
            user.name = name

            # Update password only if user entered a new one
            if new_password:
                user.set_password(new_password)

            db.session.commit()
            message = "Profile updated successfully."

    return render_template("profile.html", user=user, message=message, error=error)


# -------------------- Chat Routes -------------------- #

@app.route("/")
def home():
    user = get_current_user()
    if not user:
        # return redirect(url_for("login"))
        return render_template("welcome.html")

    chats = Chat.query.filter_by(user_id=user.id).order_by(Chat.updated_at.desc()).all()

    username = user.name if user and user.name else ""

    return render_template(
        "index.html",
        user=user,
        chats=chats,
        username=username
    )


@app.route("/get")
def get_bot_response():
    user = get_current_user()
    if not user:
        return "Unauthorized", 401

    userText = request.args.get('msg')
    print("User said:", userText)

    # 1) Get or create the active chat for this user
    chat = get_or_create_active_chat(user)

    # 2) If this is the first message and chat has no title yet → set title
    if userText and not chat.title:
        # Use first user message (trimmed) as chat title
        chat.title = userText[:80]

    # 3) Save user message
    user_msg = Message(chat_id=chat.id, sender="user", text=userText)
    db.session.add(user_msg)

    # 4) Get bot reply and save it
    bot_response_text = chatbot_response(userText)
    bot_msg = Message(chat_id=chat.id, sender="bot", text=bot_response_text)
    db.session.add(bot_msg)

    # 5) Commit all changes (chat title, messages)
    db.session.commit()

    return bot_response_text


# -------------------- Extra API routes (sidebar, history, delete) -------------------- #

@app.route("/api/chats")
def api_list_chats():
    """Return the current user's chats as JSON (for sidebar)."""
    user = get_current_user()
    if not user:
        return "Unauthorized", 401

    chats = Chat.query.filter_by(user_id=user.id).order_by(Chat.updated_at.desc()).all()
    data = []
    for c in chats:
        data.append({
            "id": c.id,
            "title": c.title or "(untitled)",
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        })
    return jsonify(data)


@app.route("/api/chat/<int:chat_id>/messages")
def api_chat_messages(chat_id):
    """Return all messages for a given chat as JSON, and mark it as active."""
    user = get_current_user()
    if not user:
        return "Unauthorized", 401

    chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
    if not chat:
        return "Chat not found", 404

    # Set this chat as the active one
    session["active_chat_id"] = chat.id

    messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
    data = []
    for m in messages:
        data.append({
            "sender": m.sender,
            "text": m.text,
            "timestamp": m.timestamp.isoformat(),
        })
    return jsonify(data)


@app.route("/api/chat/<int:chat_id>/delete", methods=["POST"])
def api_delete_chat(chat_id):
    """Delete a chat and all its messages for the current user."""
    user = get_current_user()
    if not user:
        return jsonify({"error": "unauthorized"}), 401

    chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
    if not chat:
        return jsonify({"error": "chat_not_found"}), 404

    # Delete messages explicitly (safe even if cascade is defined)
    Message.query.filter_by(chat_id=chat.id).delete()

    # Delete the chat itself
    db.session.delete(chat)
    db.session.commit()

    # If this was the active chat, clear it from the session
    if session.get("active_chat_id") == chat_id:
        session.pop("active_chat_id", None)

    return jsonify({"status": "ok"})


@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    """Create a brand new empty chat and make it active."""
    user = get_current_user()
    if not user:
        return "Unauthorized", 401

    chat = Chat(user_id=user.id, title=None)
    db.session.add(chat)
    db.session.commit()

    session["active_chat_id"] = chat.id

    return jsonify({"chat_id": chat.id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
