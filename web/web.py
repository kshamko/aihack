from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':

        text = request.form['text']
        print(text)

    return render_template('layout.html')

if __name__ == "__main__":
    app.run()