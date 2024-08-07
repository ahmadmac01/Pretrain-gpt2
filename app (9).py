import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="GPT-2 Text Generator", layout="centered")

tokenizer = AutoTokenizer.from_pretrained("ahmadmac/Pretrained-GPT2")
model = AutoModelForCausalLM.from_pretrained("ahmadmac/Pretrained-GPT2")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

st.title("GPT-2 Text Generator")
st.write("Enter a prompt to generate text using GPT-2")

user_input = st.text_input("Prompt")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating..."):
            generated_text = generate_text(user_input)
            st.write(generated_text)
    else:
        st.warning("Please enter a prompt")
