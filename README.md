## Pretrained Distill GPT-2 
This project demonstrates the process of pretraining a GPT-2 model on a dataset of Shakespeare dialogues, pushing the model to Hugging Face Hub, and deploying a Streamlit interface for text generation. The model `ahmadmac/Pretrained-GPT2` is trained on the `Trelis/tiny-shakespeare` dataset and can generate text in the style of Shakespearean dialogue.

## How It Works
1. **Dataset Preparation:**

We start by loading the dataset named "Grades.csv" from the Pandas library. This dataset contains relevant data that we will use to fine-tune the Distill GPT-2 model.
```python
csv_file = 'grades.csv'  
df = pd.read_csv(csv_file)
```
2. **Data Preprocessing:**

We preprocess the data by tokenizing the text and padding it to a fixed length. Tokenization converts the text into a format that the GPT-2 model can understand, padding ensures that all sequences are the same length for efficient processing.

3. **Model Training:**

We use the Hugging Face Trainer API to train the GPT-2 model on the tokenized dataset. The Trainer API simplifies the training process by handling various aspects such as gradient accumulation, mixed precision training, and logging.
``` python
from transformers import Trainer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
```
### [Model URL](https://huggingface.co/ahmadmac/Pretrained-GPT2)

4. **Pushing to Hugging Face Hub:**

After training, we push the model and tokenizer to the Hugging Face Hub. This makes the model accessible to others and allows for easy sharing and deployment.

```python
trainer.push_to_hub("Pretrained-GPT2")
tokenizer.push_to_hub("Pretrained-GPT2")
```

5. **Streamlit Interface:**
   
Deploy a Streamlit interface for generating text using the pretrained GPT-2 model. Streamlit provides an easy way to create web applications, allowing users to interact with the model by entering a prompt and receiving generated text in response.

6. **Deployement:**

The Streamlit app is deployed on Hugging Face Spaces, making it accessible for users to interact with the GPT-2 model trained on Shakespeare dialogues.

## Files
- **app.py:** The main Streamlit application file.
- **requirements.txt:** Lists the dependencies for the project.
## Setup
1. **Install requirements.txt file**
```python
!pip install -r requirements.txt
```
2. **Install Libraries**

```python
!pip install streamlit transformers torch
```
## Deploy the Streamlit app on Hugging Face Spaces:

- Create a new space on Hugging Face.
- Push the project files (app.py and requirements.txt) to the space repository.
- Visit the Hugging Face Space to interact with the app.
__________________________
Try it out!
## [Hugging Face Space URL](https://huggingface.co/spaces/ahmadmac/Pretrain-GPT)




