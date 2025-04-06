
import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('data/ecommerce_data.csv', encoding='ISO-8859-1')

# Initial inspection
print("\nDataset Info:")
print(df.info())

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Convert 'InvoiceDate' to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Add TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Remove canceled orders (InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative or zero quantities or prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Show cleaned dataset shape
print(f"\nCleaned Data Shape: {df.shape}")

# Save cleaned data
df.to_csv('data/cleaned_ecommerce_data.csv', index=False)

# Show top 5 products by quantity sold
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Products Sold:\n", top_products)

# Revenue by country
revenue_by_country = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
print("\nRevenue by Country:\n", revenue_by_country.head(5))

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# Load the cleaned data
cleaned_df = pd.read_csv('data/cleaned_ecommerce_data.csv')

# Create a customer-product matrix
customer_product_matrix = cleaned_df.pivot_table(index='CustomerID', 
                                                  columns='Description', 
                                                  values='Quantity', 
                                                  aggfunc='sum', 
                                                  fill_value=0)

# Apply Truncated SVD to reduce dimensions
svd = TruncatedSVD(n_components=20, random_state=42)
customer_embeddings = svd.fit_transform(customer_product_matrix)

# Normalize the embeddings
normalized_embeddings = normalize(customer_embeddings)

# Store embeddings as DataFrame
customer_embedding_df = pd.DataFrame(normalized_embeddings, 
                                     index=customer_product_matrix.index)
customer_embedding_df.reset_index(inplace=True)

# Save the embeddings to file
customer_embedding_df.to_csv('data/customer_embeddings.csv', index=False)

print(" Customer embeddings saved at: data/customer_embeddings.csv")

from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Drop NaN and get unique product descriptions
unique_products = cleaned_df['Description'].dropna().unique()

# Generate embeddings
# Generate embeddings
print(" Generating product embeddings...")
product_embeddings = model.encode(unique_products, show_progress_bar=True)

#  Reduce product embeddings to 20 dimensions to match customer embeddings
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=20, random_state=42)
reduced_product_embeddings = svd.fit_transform(product_embeddings)

#  Create DataFrame with reduced embeddings
product_embeddings_df = pd.DataFrame(reduced_product_embeddings)
product_embeddings_df['Description'] = unique_products

#  Save to CSV
product_embeddings_df.to_csv("data/product_embeddings.csv", index=False)
print(" Reduced product embeddings saved at: data/product_embeddings.csv")


# Save to CSV

import sqlite3
import pandas as pd

#  Load embedding CSVs
customer_embeddings = pd.read_csv("data/customer_embeddings.csv")
product_embeddings = pd.read_csv("data/product_embeddings.csv")
print(" Columns in customer_embeddings:")
print(customer_embeddings.columns)




#  Convert to string (because lists can't be stored directly in DB)
#  Combine columns 0 to 19 into a single embedding column
embedding_cols = [str(i) for i in range(20)]
customer_embeddings['embedding'] = customer_embeddings[embedding_cols].values.tolist()

#  Convert to string (because lists can't be stored directly in DB)
customer_embeddings['embedding'] = customer_embeddings['embedding'].apply(str)
#  Combine columns 0 to 19 into a single embedding column

#  Combine columns 0 to 19 into a single embedding column


embedding_cols = [col for col in product_embeddings.columns if col != 'Description']
product_embeddings['embedding'] = product_embeddings[embedding_cols].values.tolist()

#  Convert list to string before saving to DB
product_embeddings['embedding'] = product_embeddings['embedding'].apply(str)

conn = sqlite3.connect("genie_cart.db")  # OR "data/embeddings.db" if you prefer

# Save only necessary columns
customer_embeddings[['CustomerID', 'embedding']].to_sql(
    'customer_embeddings',
    conn,
    if_exists='replace',
    index=False
)

product_embeddings[['Description', 'embedding']].to_sql(
    'product_embeddings',
    conn,
    if_exists='replace',
    index=False
)

print(" Embeddings successfully saved to SQLite database 'genie_cart.db'")








