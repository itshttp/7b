#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for the application.
This file serves as the entry point for the program.
Uses llama.cpp to interact with the Llama 7B model.
"""

import os
import sys
print(f"Python interpreter path: {sys.executable}")
print(f"Python version: {sys.version}")
from llama_cpp import Llama

def get_model_path():
    """Get the path to the GGUF model file."""
    model_path = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model_path

def create_llama_model():
    """Create and return a Llama model instance."""
    model_path = get_model_path()
    return Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_threads=4,  # Number of CPU threads to use
        n_gpu_layers=0  # Number of layers to offload to GPU (0 for CPU-only)
    )

def generate_response(llm, prompt):
    """Generate a response for the given prompt."""
    # Format prompt for Llama-2-chat without duplicate <s> tag
    chat_prompt = f"[INST] {prompt} [/INST]"
    
    # Generate response
    response = llm(
        chat_prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        echo=False,
        stop=["</s>"]
    )
    
    return response['choices'][0]['text'].strip()

def main():
    """
    Main function that serves as the entry point of the program.
    """
    try:
        # Initialize the model
        llm = create_llama_model()
        print("Model loaded successfully!")
        
        # Example conversation
        while True:
            user_input = input("\nEnter your message (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            # Generate and print response
            response = generate_response(llm, user_input)
            print("\nAssistant:", response)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 