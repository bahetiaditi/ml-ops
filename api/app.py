from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/hello/<name>')
def index(name):
    return "Hello, "+name+"!"

@app.route('/model', methods=['POST'])
def pred_model():
    js = request.get_json()
    x= js['x']
    y = js['y']
    return x+y