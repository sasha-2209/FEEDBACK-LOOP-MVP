from google import generativeai as genai
genai.configure(api_key="AIzaSyCoDW2yklJUB74zTGpd8NQJf_ifPdEkPU0")

model = genai.GenerativeModel("gemini-2.5-pro")
resp = model.generate_content("Say 'hello world'")
print(resp.text)