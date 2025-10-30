import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

print("Attempting to configure API key...")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring API key: {e}")
    print("Make sure your GOOGLE_API_KEY is correct in the .env file.")
    exit()

print("--- Available Models for your API key ---")

try:
    # This is the "Call ListModels" part
    for m in genai.list_models():
        # Check if the model supports the 'generateContent' method
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
            print(f"   - Display Name: {m.display_name}")
            print(f"   - Supports: {m.supported_generation_methods}\n")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to list models: {e}")
    print("This could be an API key issue or a network/firewall problem.")

print("------------------------------------------")