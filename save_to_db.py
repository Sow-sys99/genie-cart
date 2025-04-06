import pandas as pd
import sqlite3

# Step 1: Load the customer and product embeddings CSV files
customer_embeddings = pd.read_csv("data/customer_embeddings.csv")
product_embeddings = pd.read_csv("data/product_embeddings.csv")

# Step 2: Combine all embedding columns into a list in a new column called 'embedding'
# For customers
customer_embedding_cols = [col for col in customer_embeddings.columns if col.startswith("embedding")]
customer_embeddings['embedding'] = customer_embeddings[customer_embedding_cols].values.tolist()
customer_embeddings['embedding'] = customer_embeddings['embedding'].apply(str)

# For products
product_embedding_cols = [col for col in product_embeddings.columns if col.startswith("embedding")]
product_embeddings['embedding'] = product_embeddings[product_embedding_cols].values.tolist()
product_embeddings['embedding'] = product_embeddings['embedding'].apply(str)

# Step 3: Save to SQLite database
conn = sqlite3.connect("genie_cart.db")

# Save customer embeddings (only CustomerID and embedding)
customer_embeddings[['CustomerID', 'embedding']].to_sql("customer_embeddings", conn, if_exists="replace", index=False)

# Save product embeddings (only Description and embedding)
product_embeddings[['Description', 'embedding']].to_sql("product_embeddings", conn, if_exists="replace", index=False)

print("✅ Embeddings successfully saved to SQLite database.")
conn.close()
