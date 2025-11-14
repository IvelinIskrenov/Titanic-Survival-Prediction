from flask import Flask, jsonify, render_template
from TitanicSurvivalPredictionsModel import SurvivalPrediction

app = Flask(__name__)

model = SurvivalPrediction()
trained = False


@app.route('/')
def home():
    return render_template('index.html'), 200

@app.route('/train')
def train():
    global trained
    try:
        model.load_data()
        model.split_data()
        model.preprocessing()
        model.train_RF()
        trained = True
        #something should appear
        return jsonify({"status": "trained successfully"}),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    #return jsonify({"trained": trained})
    return render_template('status.html', trained=trained), 200

@app.route('/evaluate')
def evaluate():
    if not trained:
        return jsonify({"error": "Model not trained yet"}), 400
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
