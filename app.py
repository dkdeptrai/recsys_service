from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation import recommend_shoes

app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Endpoint to recommend shoes based on a given shoe ID.
    """
    data = request.get_json()
    shoe_id = data['shoe_id']
    recommendations = recommend_shoes(shoe_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
