import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def analyze_code(code_snippet):
    inputs = tokenizer.encode(code_snippet, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150)
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return feedback

if __name__ == '__main__':
    code_example = '''
# Your code example here
def example_function(x):
    return x + 1
'''
    print(analyze_code(code_example))
