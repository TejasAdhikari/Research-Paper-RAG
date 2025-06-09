import os
import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()  


# Configure logging
logger = logging.getLogger(__name__)


class GeminiLLM:
    # Initialize Gemini with API key from environment
    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY environment variable required!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini LLM initialized")
    

    # Generate response from Gemini
    def generate_response(self, prompt, max_tokens=2000):
        try:
            # Create generation configuration
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            )
            # Generate response using the model
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    

    # Test if Gemini is working
    def test_connection(self):
        try:
            response = self.generate_response("Say 'Hello, I am working correctly!'")
            return "working correctly" in response.lower()
        except:
            return False


# Test the Gemini wrapper
if __name__ == "__main__":
    try:
        llm = GeminiLLM()
        
        if llm.test_connection():
            print("✅ Gemini connection successful!")
            
            # Test with a simple prompt
            response = llm.generate_response("Explain machine learning in one sentence.")
            print(f"Response: {response}")
        else:
            print("❌ Gemini connection failed!")
            
    except Exception as e:
        print(f"❌ Setup error: {e}")