import requests
import os

def download_model():
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    output_path = os.path.join("C:", "Users", "itsht", "LLM_models", "llama-2-7b-chat.Q4_K_M.gguf")
    
    # Create LLM_models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Downloading model... This may take a while...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Model downloaded successfully to {output_path}")

if __name__ == "__main__":
    download_model() 