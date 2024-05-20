import pandas as pd

xls = pd.ExcelFile('./modified_shoes_new (2).xlsx')

df = xls.parse(xls.sheet_names[0])

df['images_list'] = df['images_list'].str.split(',')

df_exploded = df.explode('images_list')

selected_columns = df_exploded[['id', 'images_list']].rename(columns={'images_list': 'image_url', "id": "product_id"})

if 'id' not in selected_columns.columns:
    selected_columns['id'] = range(1, len(selected_columns) + 1)

selected_columns = selected_columns.reindex(columns=['id', 'product_id', 'image_url'])
selected_columns['image_url'] = selected_columns['image_url'].str.strip()

selected_columns.to_csv('./shoes_images.csv', index=False)
