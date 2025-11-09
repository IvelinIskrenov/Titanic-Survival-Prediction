from flask import Flask, jsonify, render_template
from TitanicSurvivalPredictionsModel import SurvivalPrediction

app = Flask(__name__)

model = SurvivalPrediction()
trained = False


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train():
    global trained
    try:
        model.load_data()
        model.split_data()
        model.preprocessing()
        model.train_RF()
        trained = True
        return jsonify({"status": "trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    return jsonify({"trained": trained})

@app.route('/evaluate')
def evaluate():
    if not trained:
        return jsonify({"error": "Model not trained yet"}), 400
    results = model.evaluate()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
