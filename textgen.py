from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=100):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set pad_token to eos_token to suppress warnings
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate output sequence
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ✅ Now using attention mask
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print("\nGenerated text:\n")
    print(generate_text(prompt))
