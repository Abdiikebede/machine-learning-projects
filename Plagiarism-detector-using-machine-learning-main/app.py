from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer (will run once when app starts)
print("Loading model and vectorizer...")
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
print("Model loaded successfully! Ready to catch plagiarists")

def detect(input_text):
    if not input_text.strip():
        return "Error: Please enter some text!"
    vectorized = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(vectorized)[0]
    return "Plagiarism Detected" if prediction == 1 else "No Plagiarism Detected"

# Single route that handles BOTH GET and POST
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_text = request.form.get('text', '')
        result = detect(user_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
