
from utils import get_index
from data_processing import load_data

df, final_similarity = load_data()

def recommend_shoes(shoe_id, num_recommendations=10, sim_matrix=final_similarity, least_similar=False, page = 1, page_size = 10, get_all = False):
    """
    
    Parameters:
    - shoe_id: The ID of the shoe for which recommendations are to be made.
    - sim_matrix: The similarity matrix used to find similar shoes.
    - num_recommendations: The number of recommendations to return.
    - least_similar: If True, recommend the least similar shoes; otherwise, recommend the most similar shoes.
    - page: The page number of the recommendations.
    - page_size: The number of recommendations to return in each page.
    
    Returns:
    - A list of recommended shoes, each represented as a dictionary with 'asin' and 'score' keys.
    """
    idx = get_index(df, shoe_id)
    
    # Get the pairwise similarity scores of all shoes with that shoe
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort the scores based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=not least_similar)
    if(get_all): return sim_scores

    total_pages = (len(sim_scores) // page_size) + (len(sim_scores) % page_size!= 0)

    # Pagination
    start = (page - 1) * page_size
    end = start + page_size
    if len(sim_scores) < end:
        end = len(sim_scores)
        sim_scores = sim_scores[start + 1:end]
    else:
        sim_scores = sim_scores[start + 1:end + 1]


    # Convert the indices and scores to a list of dictionaries
    shoe_recommendations = [{'asin': df['asin'].iloc[i[0]], 'score': i[1]} for i in sim_scores]
    
    return shoe_recommendations, total_pages

def recommend_from_last_viewed_items(shoes_ids, page=1, page_size=10):
    """
    Recommend shoes based on the last viewed items with pagination support.
    
    Parameters:
    - shoes_ids: The IDs of the shoes the user has viewed.
    - page: The page number of the recommendations.
    - page_size: The number of recommendations to return in each page.
    
    Returns:
    - A paginated list of recommended shoes.
    """
    recommendations = []
    for shoe_id in shoes_ids:
        recs = recommend_shoes(shoe_id, get_all=True)
        recommendations.extend(recs)
    
    # Sort the recommendations based on the similarity score
    sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    unique_recommendations = []
    seen_asins = set()

    for rec in sorted_recommendations:
        if rec[0] not in seen_asins:
            unique_recommendations.append(rec)
            seen_asins.add(rec[0])


    total_pages = (len(unique_recommendations) // page_size) + (len(unique_recommendations) % page_size!= 0)
    
    # Slice unique_recommendations based on page
    start = (page - 1) * page_size
    end = min(start + page_size, len(unique_recommendations))
    unique_recommendations = unique_recommendations[start:end]

    shoe_recommendations = [{'asin': df['asin'].iloc[i[0]], 'score': i[1]} for i in unique_recommendations]
    
    return shoe_recommendations, total_pages
