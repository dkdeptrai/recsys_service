
from utils import get_index
from data_processing import load_data

df, final_similarity = load_data()

def recommend_shoes(shoe_id, num_recommendations=10, sim_matrix=final_similarity, least_similar=False):
    """
    Recommend shoes based on the similarity matrix.
    
    Parameters:
    - shoe_id: The ID of the shoe for which recommendations are to be made.
    - sim_matrix: The similarity matrix used to find similar shoes.
    - num_recommendations: The number of recommendations to return.
    - least_similar: If True, recommend the least similar shoes; otherwise, recommend the most similar shoes.
    
    Returns:
    - A list of recommended shoes, each represented as a dictionary with 'asin' and 'score' keys.
    """
    idx = get_index(df, shoe_id)
    
    # Get the pairwise similarity scores of all shoes with that shoe
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort the scores based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=not least_similar)
    
    # Select the top N recommendations
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Convert the indices and scores to a list of dictionaries
    shoe_recommendations = [{'asin': df['asin'].iloc[i[0]], 'score': i[1]} for i in sim_scores]
    
    return shoe_recommendations

def recommend_from_last_viewed_items(shoes_ids, num_recommendations_each = 2):
    """
    Recommend shoes based on the similarity matrix.
    
    Parameters:
    - shoes_ids: The IDs of the shoes the user has viewed.
    - num_recommendations_each: The number of recommendations to return for each shoe.
    
    Returns:
    - A list of recommended shoes, each represented as a dictionary with 'asin' and 'score' keys.
    """
    recommendations = []
    for shoe_id in shoes_ids:
        recommendations += recommend_shoes(shoe_id, num_recommendations=num_recommendations_each)
    
    return recommendations
