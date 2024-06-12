import pandas as pd
import random

# Load the original CSV file
df = pd.read_csv('./updated_product_with_brands.csv')

# Predefined lists for colors and stock counts
colors = ["#5E6167", "#312951", "#FFFFFF", "#000000", "#FF0000", "#FFD700"]
stock_counts = list(range(10, 201, 10))  # Stock counts from 10 to 200 in steps of 10

# Manually generate the list of sizes from 4.5 to 9.0 in increments of 0.5
sizes = [i * 0.5 for i in range(9, 18)]

# List to hold rows for the output DataFrame
rows = []

# Iterate over each product
for index, row in df.iterrows():
    # Iterate through each size
    for size in sizes:
        # Round the size to one decimal place for consistency
        size = round(size, 1)
        
        # Randomly select a color and stock count for the current size
        color = random.choice(colors)
        stock = random.choice(stock_counts)
        
        # Add a new row to the list
        rows.append({
            'id': len(rows) + 1,
            'shoes_id': row['id'],
            'size': str(size),
            'color': color,
            'stock': stock
        })

# Convert the list of rows to a DataFrame
output_df = pd.DataFrame(rows)

# Optionally, convert specific columns to integer type if needed
output_df[['id', 'shoes_id', 'stock']] = output_df[['id', 'shoes_id', 'stock']].astype(int)

# Save the output DataFrame to a new CSV file
output_df.to_csv('shoe_specifications.csv', index=False)
