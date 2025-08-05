from flask import Flask, request, jsonify
from waitress import serve
from modal_loader import load_model  # this should load English model only

app = Flask(__name__)

# Load model and vectorizer once at server startup
print("🔄 Loading punjabi spam detection model...")
vectorizer, model = load_model()
print("✅ Model loaded successfully.")

def check_spam_pa(text):
    if not text:
        return None
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec.toarray())
        print(pred)
        return bool(pred[0])
    except Exception as e:
        print(f"Error checking spam: {e}")
        return None

@app.route("/check-spam", methods=["POST"])
def check_en():
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "Please provide a text"}), 400
        
        is_spam = check_spam_pa(text)

        if is_spam is None:
            return jsonify({"error": "Unable to process text"}), 500

        return jsonify({"isSpam": is_spam}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Spam detection service running on http://0.0.0.0:8080")
    serve(app, host="0.0.0.0", port=8080)
