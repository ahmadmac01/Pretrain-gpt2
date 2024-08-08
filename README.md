## Pretrained Distill GPT-2 
This project involves training the DistilGPT-2 language model on a custom Question & Answer (Q&A) dataset generated from a CSV file containing grades. The model is trained to understand and generate specific answers based on queries related to the students' data, such as first names, Social Security Numbers (SSNs), test scores, and final grades. The fine-tuned model is then pushed to Hugging Face and deployed using a Streamlit interface.

## How It Works
1. Loading the Dataset
We start by loading the Grades.csv file using the pandas library. This CSV file contains various columns such as First name, Last name, SSN, Test1, Grade, etc.

```python
csv_file = 'grades.csv'  
df = pd.read_csv(csv_file)
```
2. **Creating Q&A Pairs**
For each row in the dataset, we generate multiple Q&A pairs. Each pair consists of a question related to the student's data and the corresponding answer. This helps in creating a diverse set of training examples for the model.

3. **Preprocessing the Data**

The data is tokenized using the DistilGPT-2 tokenizer, ensuring that the inputs and outputs are properly padded and truncated for model compatibility.

4. **Model Training:**

The DistilGPT-2 model is fine-tuned using the tokenized Q&A dataset. Training configurations, such as learning rate, batch size, and number of epochs, are defined and passed to the `Trainer class` for training.
``` python
from transformers import Trainer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
```
### [Model URL](https://huggingface.co/ahmadmac/DistillGPT2-CSV)

5. **Pushing to Hugging Face Hub:**

After training, we push the model and tokenizer to the Hugging Face Hub. This makes the model accessible to others and allows for easy sharing and deployment.

```python
trainer.push_to_hub("ahmadmac/DistillGPT2-CSV")
tokenizer.push_to_hub("ahmadmac/DistillGPT2-CSV")
```

5. **Streamlit Interface:**
   
Deploy a Streamlit interface for generating text using the pretrained Distill GPT-2 model. Streamlit provides an easy way to create web applications, allowing users to interact with the model by entering a prompt and receiving generated text in response.

6. **Deployement:**

We use Streamlit to create an interactive interface for the model, where users can input prompts and receive generated text from the fine-tuned DistilGPT-2 model.

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




