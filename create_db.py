import sqlite3
import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_ecommerce_data.csv")

# Connect to the SQLite database
conn = sqlite3.connect("db/genie_cart.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    StockCode TEXT PRIMARY KEY,
    Description TEXT,
    UnitPrice REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    CustomerID REAL PRIMARY KEY,
    Country TEXT
)
""")

# Insert into tables
product_data = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
product_data.to_sql('products', conn, if_exists='replace', index=False)

customer_data = df[['CustomerID', 'Country']].drop_duplicates().dropna(subset=['CustomerID'])
customer_data.to_sql('customers', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(" SQLite database created and data inserted successfully!")
