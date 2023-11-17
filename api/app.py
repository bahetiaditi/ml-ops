from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def pred_model():
    js = request.get_json()
    image_path_1 = js['image1']
    image_path_2 = js['image2']
   
    model = load('Models/tree_max_depth:10.joblib')
    prediction_1 = model.predict(image_path_1)
    prediction_2 = model.predict(image_path_2)
    are_same = np.array_equal(prediction_1, prediction_2)

    return jsonify({'result': are_same})