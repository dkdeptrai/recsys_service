from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation import recommend_shoes, recommend_from_last_viewed_items

app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Endpoint to recommend shoes based on a given shoe ID.
    """
    data = request.get_json()
    shoe_id = data['shoe_id']
    num_of_recommendations = data['num_of_recommendations'] if 'num_of_recommendations' in data else 10
    recommendations = recommend_shoes(shoe_id, num_recommendations=num_of_recommendations)
    return jsonify(recommendations)

@app.route('/recommend_from_user_last_viewed_items', methods=['POST'])
def recommend_from_last_viewed():
    """
    Endpoint to recommend shoes based on the user's last viewed items.
    """
    data = request.get_json()
    shoes_ids = data['shoes_ids']
    num_of_recommendations_for_each = data['num_of_recommendations_for_each']
    recommendations = recommend_from_last_viewed_items(shoes_ids, num_of_recommendations_for_each)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
