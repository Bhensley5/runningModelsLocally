import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Authenticate with Hugging Face
hf_token = "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"  # replace with your actual token
login(hf_token)

# Define the paths to save the model and tokenizer
model_path = './Meta-Llama-3-8B-Instruct'
tokenizer_path = './Meta-Llama-3-8B-Instruct-tokenizer'

# Function to load the model and tokenizer efficiently
def load_model_and_tokenizer(model_path, tokenizer_path, hf_token):
    if not os.path.exists(model_path):
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', token=hf_token)
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token

        # Load the model with FP16
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', token=hf_token, torch_dtype=torch.float16)
        
        # Save the tokenizer and model locally
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
        print("Model and tokenizer saved locally.")
    else:
        # Load the tokenizer from local disk
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token

        # Load the model from local disk with FP16
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        print("Model and tokenizer loaded from local disk.")
    
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, hf_token)

# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Function to generate text based on a prompt using FP16 with custom stopping logic
def generate_text(prompt, max_length=2000, temperature=0.7, top_p=0.9, top_k=120, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    torch.cuda.empty_cache()  # Clear unused memory
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id # Add a length penalty to discourage very long sequences
        )
    
    # Decode the generated tokens
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

# Loop to enter prompts until 'STOP' is entered
while True:
    prompt = input("Enter a prompt (type 'STOP' to end): ")
    if prompt.strip().upper() == 'STOP':
        break
    generated_text = generate_text(prompt)
    print("Generated text:", generated_text)
