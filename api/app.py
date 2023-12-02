from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)

@app.route('/hello/<name>')
def index(name):
    return "Hello, "+name+"!"

@app.route('/predict', methods=['POST'])
def pred_model():
    js = request.get_json()
    image1 = js['image']
    #Assuming this is the path of our best trained model
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../models/treemax_depth:100.joblib')
    model = load(filename)
    pred1 = model.predict(image1)
    #reurn pred1 in json
    return jsonify(prediction=pred1.tolist())




def load_models():
    models = {}

    # Load SVM model
    svm_path = 'models/svmgamma:0.001_C:1.joblib'
    models['svm'] = load(svm_path) 

    # Load Logistic Regression model
    lr_path = 'm23csa001_lr_lbfgs.joblib'
    models['lr'] = load(lr_path) 

    # Load Decision Tree model
    tree_path = 'models/treemax_depth:10.joblib'
    models['tree'] = load(tree_path) 

    return models

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    models = load_models()  # Load all models
    model = models.get(model_type)

    if model is None:
        return jsonify({"error": f"Model type '{model_type}' not found."}), 404

    js = request.get_json()
    image = js['image']
    pred = model.predict(image)
    return jsonify(prediction=pred.tolist())



if __name__ == '__main__':
    app.run(debug=True)
    