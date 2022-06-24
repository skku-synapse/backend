from flask import Flask, jsonify
from flask_cors import CORS

api = Flask(__name__)
CORS(api)

@api.route('/')
def my_profile():
    response_body = {
        "name": "Jinsuk",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body


api.run(host='0.0.0.0', port=5001, debug=True)
 