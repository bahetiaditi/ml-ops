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

def test_predict_svm(client):
    test_data = {'image': ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}
    response = client.post('/predict/svm', json=test_data)
    assert response.status_code == 200

def test_predict_lr(client):
    test_data = {'image': ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}
    response = client.post('/predict/linear_regression', json=test_data)
    assert response.status_code == 200

if __name__ == '__main__':
    app.run(debug=True)