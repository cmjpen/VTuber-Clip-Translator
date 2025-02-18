from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
from main import main, test

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    youtube_url = data.get("url")
    
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400
    
    try:
        result = test(youtube_url)

        if result:
            return jsonify({"message": result["message"]})
        else:
            return jsonify({"error": "Processing failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
