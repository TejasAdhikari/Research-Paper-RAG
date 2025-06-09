import os
import google.generativeai as genai

# Test API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ GOOGLE_API_KEY not set!")
else:
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello! Please respond with 'API working correctly.'")
        print(f"✅ Gemini response: {response.text}")
    except Exception as e:
        print(f"❌ API error: {e}")