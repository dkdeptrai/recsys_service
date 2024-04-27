import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from utils import binary

def load_data():
    """
    Load the shoes data from a CSV file, drop rows with missing values, and prepare the data for processing.
    """
    df = pd.read_csv('./Data/shoes.csv')
    df = df.dropna()
    
    # Processing outer material column
    outerMaterialsList = list(df['Outer Material'].unique())
    
    # Processing breadcrumbs column
    df.rename(columns={'breadcrumbs': 'Categories'}, inplace=True)
    df['Categories'] = df['Categories'].apply(ast.literal_eval)
    
    # Processing Inner Material column
    innerMaterialsList = list(df['Inner Material'].unique())
    
    # Processing Sole Column
    soleMaterialsList = list(df['Sole'].unique())
    
    # Processing Heel Type Column
    heelTypesList = list(df['Heel Type'].unique())
    
    # Processing Shoe Width Column
    shoeWidthsList = list(df['Shoe Width'].unique())
    
    # Processing Closure Column
    closuresList = list(df['Closure'].unique())
    
    # Processing Brand Column
    brandsList = list(df['brand'].unique())
    
    # Creating the binary columns
    df['Outer Material bin'] = df['Outer Material'].apply(lambda x: binary(x, outerMaterialsList))
    df['Categories bin'] = df['Categories'].apply(lambda x: binary(x, list(df['Categories'].explode().unique())))
    df['Inner Material bin'] = df['Inner Material'].apply(lambda x: binary(x, innerMaterialsList))
    df['Sole bin'] = df['Sole'].apply(lambda x: binary(x, soleMaterialsList))
    df['Heel Type bin'] = df['Heel Type'].apply(lambda x: binary(x, heelTypesList))
    df['Shoe Width bin'] = df['Shoe Width'].apply(lambda x: binary(x, shoeWidthsList))
    df['Closure bin'] = df['Closure'].apply(lambda x: binary(x, closuresList))
    df['Brand bin'] = df['brand'].apply(lambda x: binary(x, brandsList))
    
    # Convert the binary vectors to numpy arrays
    Categories_bin = np.array(df['Categories bin'].tolist())
    OuterMaterial_bin = np.array(df['Outer Material bin'].tolist())
    InnerMaterial_bin = np.array(df['Inner Material bin'].tolist())
    Sole_bin = np.array(df['Sole bin'].tolist())
    Closure_bin = np.array(df['Closure bin'].tolist())
    HeelType_bin = np.array(df['Heel Type bin'].tolist())
    ShoeWidth_bin = np.array(df['Shoe Width bin'].tolist())
    Brand_bin = np.array(df['Brand bin'].tolist())
    
    # Initialize an empty dictionary to store the similarity matrices
    similarity_matrices = {}
    
    # List of category names
    category_names = ['Categories_bin','OuterMaterial_bin', 'InnerMaterial_bin', 'Sole_bin', 'Closure_bin', 'HeelType_bin', 'ShoeWidth_bin', 'Brand_bin']
    
    # Loop through each category
    for i, bin_col in enumerate([Categories_bin, OuterMaterial_bin, InnerMaterial_bin, Sole_bin, Closure_bin, HeelType_bin, ShoeWidth_bin, Brand_bin]):
        # Convert the binary vectors to numpy arrays
        bin_vec = np.array(bin_col.tolist())
        
        # Calculate the cosine similarity between items
        cosine_sim = cosine_similarity(bin_vec)
        
        # Store the cosine similarity matrix in the dictionary
        similarity_matrices[category_names[i]] = cosine_sim
    
    # Apply weights to the similarity matrices
    weights = {'Categories_bin': 0.2, 'OuterMaterial_bin': 0.1, 'InnerMaterial_bin': 0.1, 'Sole_bin': 0.1, 'Closure_bin': 0.1, 'HeelType_bin': 0.1, 'ShoeWidth_bin': 0.1, 'Brand_bin': 0.2}
    final_similarity = np.zeros((len(df), len(df)))
    
    for category, weight in weights.items():
        weighted_similarity = similarity_matrices[category] * weight
        final_similarity += weighted_similarity
    
    return df, final_similarity
