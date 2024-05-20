from flask import Flask, request, jsonify
import json
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
    page = data['page'] if 'page' in data else 1
    page_size = data['page_size'] if 'page_size' in data else 10

    num_of_recommendations = data['num_of_recommendations'] if 'num_of_recommendations' in data else 10
    recommendations, total_pages = recommend_shoes(shoe_id, num_recommendations=num_of_recommendations, page = page, page_size = page_size)
    
    return jsonify({"recommendations": recommendations, "total_pages": total_pages})

@app.route('/recommend_from_user_last_viewed_items', methods=['POST'])
def recommend_from_last_viewed():
    """
    Endpoint to recommend shoes based on the user's last viewed items.
    """
    data = request.get_json()
    shoes_ids = data['shoes_ids']
    page = data['page'] if 'page' in data else 1
    page_size = data['page_size'] if 'page_size' in data else 10
    recommendations, total_pages = recommend_from_last_viewed_items(shoes_ids, page=page, page_size=page_size)
    
    return jsonify({"recommendations": recommendations, "total_pages": total_pages})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
