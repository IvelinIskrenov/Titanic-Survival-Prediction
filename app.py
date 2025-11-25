from flask import Flask, jsonify, render_template, request
from TitanicSurvivalPredictionsModel import SurvivalPrediction

app = Flask(__name__)

model = SurvivalPrediction()
trainedLR = False
trainedRF = False
preprocessing_model = False

@app.route('/')
def home():
    return render_template('index.html'), 200

@app.route('/train/<model_type>')
def train_model(model_type):
    global trainedLR, trainedRF, preprocessing_model

    if model_type not in ["LR", "RF"]:
        return jsonify({"error": "Invalid model type. Must be 'LR' or 'RF'."}), 400

    try:
        if preprocessing_model != True:
            model.load_data()
            model.split_data()
            model.preprocessing()
            preprocessing_model = True

        if model_type == "LR":
            model.train_logistic_regression()
            trainedLR = True
        elif model_type == "RF":
            model.train_RF()
            trainedRF = True

        return render_template('status.html', trainedLR=trainedLR, trainedRF=trainedRF), 200 #jsonify({"status": f"{model_type} trained successfully"}), 200
    
    
    except Exception as e:
        return jsonify({"error": f"Training of {model_type} failed: {str(e)}"}), 500

@app.route('/status')
def status():
    #return jsonify({"trained": trained})
    return render_template('status.html', trainedLR=trainedLR, trainedRF=trainedRF), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not (trainedLR or trainedRF):
        return jsonify({"error": "No trained model available!"}), 400

    try:
        form_data = {
            'pclass': request.form['pclass'], 
            'sex': request.form['sex'], 
            'age': request.form['age'], 
            'sibsp': request.form['sibsp'], 
            'parch': request.form['parch'], 
            'fare': request.form['fare'], 
            'class': request.form['class'], 
            'who': request.form['who'], 
            'adult_male': request.form['adult_male'], 
            'alone': request.form['alone']
        }
        
        #passing the values as a list in the correct order
        features_order = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
        data_list = [form_data[f] for f in features_order]
        
        prediction_result = model.single_predict(data_list)
        
        if prediction_result.get("error"):
             return jsonify(prediction_result), 400
             
        return render_template('predict.html', predict_result=prediction_result) #jsonify(prediction_result), 200

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
