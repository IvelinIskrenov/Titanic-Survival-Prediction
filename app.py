from flask import Flask, jsonify, render_template, request
from TitanicSurvivalPredictionsModel import SurvivalPrediction

app = Flask(__name__)

model = SurvivalPrediction()
trainedLR = False
trainedRF = False

@app.route('/')
def home():
    return render_template('index.html'), 200

@app.route('/train/LR')
def train_LR():
    global trainedLR
    try:
        model.load_data()
        model.split_data()
        model.preprocessing()
        model.train_logistic_regression()
        trainedLR = True
        return jsonify({"status": "trained successfully"}),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train/RF')
def train_RF():
    global trainedRF
    try:
        model.load_data()
        model.split_data()
        model.preprocessing()
        model.train_RF()
        trainedRF = True
        return jsonify({"status": "trained successfully"}),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    #return jsonify({"trained": trained})
    return render_template('status.html', trainedLR=trainedLR, trainedRF=trainedRF), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not (trainedLR or trainedRF):
        return jsonify({"error": "No trained model available!"}), 400

    try:
        items = request.form.getlist("items")
        prediction = model.single_predict(items)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate')
def evaluate():
    if not (trainedLR or trainedRF):
        return jsonify({"error": "No trained model yet"}), 400

    try:
        results = model.evaluate()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html'), 200

@app.route('/help')
def help():
    return render_template('help.html'), 200

if __name__ == '__main__':
    app.run(debug=True)
