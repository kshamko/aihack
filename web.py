from flask import Flask
from flask import render_template
from flask import request
import lib

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():

    text = ""
    author = None

    if request.method == 'POST':
        text = request.form['text']

        svm = lib.get_svm()
        vectorizer, x, y, y1 = lib.get_training_set()
        text1 = lib.clean_text(text)
        x = vectorizer.transform([text1]).toarray()
        author = svm.predict(x)


    return render_template('layout.html', text = text, author = author)

if __name__ == "__main__":
    app.run()