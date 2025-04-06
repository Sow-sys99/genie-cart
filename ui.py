import streamlit as st
from agents.recommender_agent import RecommenderAgent
import requests

# Initialize agent
agent = RecommenderAgent()

# Title and instructions
st.title("🛍️ Genie Cart - Smart Product Recommender")
st.write("Select a customer and get personalized product recommendations + AI explanation!")

# Customer ID selection
customer_ids = agent.customer_embeddings['CustomerID'].dropna().unique()
customer_id = st.selectbox("Select Customer ID", customer_ids)

# Run recommender on button click
if st.button("Get Recommendations"):
    recommendations = agent.recommend(float(customer_id))

    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.subheader(" Top Product Recommendations:")
        st.dataframe(recommendations[['Description', 'similarity']])

        # Generate explanation with LLaMA 3
        product_list = recommendations['Description'].tolist()
        prompt = f"Give a friendly explanation for why a user might like the following products: {product_list}. Keep it short, fun, and personalized."

        st.subheader(" Why these products?")
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=120
                
            )
            explanation = response.json()["response"]
            st.write(explanation.strip())
        except Exception as e:
            st.error(f"Error calling LLaMA 3: {e}")
