from flask import Flask, jsonify, render_template
from TitanicSurvivalPredictionsModel import SurvivalPrediction

app = Flask(__name__)

model = SurvivalPrediction()
trained = False


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
