from gemini_llm import GeminiLLM


# Test Gemini with research paper context
def test_with_context():
    llm = GeminiLLM()
    
    # Sample context (like what you'd get from vector search)
    context = """
    Paper: "Attention Is All You Need"
    Section: Abstract
    Content: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
    """
    
    question = "What is the main contribution of this paper?"
    
    prompt = f"""Based on the following research paper context, answer the question:

                Context:
                {context}

                Question: {question}

                Answer based only on the provided context:"""
    
    response = llm.generate_response(prompt)
    print(f"Question: {question}")
    print(f"Answer: {response}")

if __name__ == "__main__":
    test_with_context()