import csv
import re

def extract_department_info(description):
    target = "Department: \\n"
    length = len(target)
    pos = description.find(target)
    
    if pos!= -1:
        info = description[(pos + length):].strip()
        return info
    else:
        return ""

def write_to_categories_table(categories, filename):
    with open(filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'category'])  # Writing headers
        for i, category in enumerate(set(categories), start=1):  # Using set to remove duplicates
            writer.writerow([i, category])

# Step 1: Extract unique categories from the original file
categories = []
with open('product.csv', mode='r', encoding='utf-8') as product_file:
    product_reader = csv.DictReader(product_file)
    for row in product_reader:
        category = extract_department_info(row['description'])
        if category:
            categories.append(category)

# Step 2: Write unique categories to the categories table
write_to_categories_table(categories, 'categories.csv')

# Step 3: Map categories to IDs and update the original file
modified_rows = []  # List to hold modified rows with category_id
with open('product.csv', mode='r', encoding='utf-8') as product_file, \
     open('updated_product.csv', mode='w', encoding='utf-8', newline='') as output_file:
    product_reader = csv.DictReader(product_file)
    output_writer = csv.DictWriter(output_file, fieldnames=product_reader.fieldnames + ['category_id'])
    
    # Write the header
    output_writer.writeheader()
    
    # Create a mapping of categories to IDs
    category_mapping = {}
    with open('categories.csv', mode='r', encoding='utf-8') as categories_file:
        categories_reader = csv.reader(categories_file)
        next(categories_reader)  # Skip the header
        for row in categories_reader:
            category_mapping[row[1]] = int(row[0])
    
    # Process each row
    for row in product_reader:
        category = extract_department_info(row['description'])
        if category:
            row['category_id'] = category_mapping.get(category, '')  # Use the mapped ID or an empty string if not found
            modified_rows.append(row)  # Append the modified row to the list
    
    # Write the modified rows to the output file
    output_writer.writerows(modified_rows)

print("Processing completed.")
