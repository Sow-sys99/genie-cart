from agents.recommender_agent import RecommenderAgent
import requests

# Function to call LLaMA 3
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
        return f" Error calling LLaMA 3: {e}"

# Initialize the recommender
agent = RecommenderAgent()

# Show available Customer IDs
print("\n Available Customer IDs:")
print(agent.customer_embeddings['CustomerID'].head())

# Ask user to input a Customer ID
customer_id = input("\n Enter a Customer ID from above list: ")

# Run recommendation
recommendations = agent.recommend(float(customer_id))
if isinstance(recommendations, str):
    print(" ", recommendations)
else:
    print("\n Top Product Recommendations:")
    print(recommendations)

    # Prepare input for LLaMA 3 reasoning
    product_list = recommendations['Description'].tolist()
    prompt = f"Give a friendly explanation for why a user might like the following products: {product_list}. Keep it short, fun, and personalized."

    print("\n Sending to LLaMA 3 for natural explanation...\n")
    explanation = call_llama3(prompt)
    print("\n LLaMA 3 says:\n")
    print(explanation.strip())
