import streamlit as st
import pandas as pd
from agents.recommender_agent import RecommenderAgent
import requests

# Page setup
st.set_page_config(page_title="Genie Cart", layout="centered")
st.title("🛒 Genie Cart - Personalized Shopping Assistant")

# Load RecommenderAgent
agent = RecommenderAgent()

# Show available customer IDs
st.subheader("Select a Customer ID")
customer_ids = agent.customer_embeddings['CustomerID'].dropna().unique().tolist()
customer_id = st.selectbox("Choose a customer", customer_ids)

# Button to run recommendation
if st.button("✨ Recommend Products"):
    with st.spinner("Generating recommendations..."):
        recommendations = agent.recommend(float(customer_id))
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.success("Here are your top recommendations:")
            st.subheader("🛍️ Recommended Products:")
            st.write("Available columns:", recommendations.columns.tolist())  # Debug line to see what's in the DataFrame

# Show whatever columns are available
            st.dataframe(recommendations)

            # Prepare prompt for LLaMA
            product_list = recommendations['Description'].tolist()
            prompt = f"Give a friendly explanation for why a user might like the following products: {product_list}. Keep it short, fun, and personalized."

            def call_llama3(prompt):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "llama3", "prompt": prompt, "stream": False},
                        timeout=60
                    )
                    response.raise_for_status()
                    return response.json()["response"]
                except requests.exceptions.RequestException as e:
                    return f"❌ Error calling LLaMA 3: {e}"

            with st.spinner("Asking Genie..."):
                explanation = call_llama3(prompt)
                st.subheader("🧞‍♂️ Why these products?")
                st.write(explanation.strip())
