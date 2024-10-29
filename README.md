# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



### Model Details

### Model Description

This model is a fine-tuned version of Llama-2-7b-chat, optimized for tasks related to fitness assistance. It has been trained to provide recommendations, answer questions, and perform related language-based tasks within the fitness and exercise domain.

- **Developed by:** Soorya03
- **Finetuned from model:** meta-llama/Llama-2-7b-chat-hf
- **Model Type:** Causal Language Model with LoRA fine-tuning
- **Language(s):** English
- **License:** Refer to the original model’s license
- **Model Repository:** Soorya03/Llama-2-7b-chat-finetune


## Uses

### Direct Use
This model is intended for interactive fitness and exercise assistance, such as providing exercise recommendations, suggesting workout routines, and answering general fitness-related questions.

### Downstream Use
May be adapted to various other fitness or health-oriented conversational applications.

### Out-of-Scope Use
Not suitable for medical or professional health advice. Avoid use cases where specialized knowledge or regulated health guidelines are required.

### Bias, Risks, and Limitations
- **Potential Bias:** The model was fine-tuned on a limited dataset and might not cover all fitness-related questions with cultural or demographic sensitivity.
- **Limitations:** Not a replacement for professional medical advice.
### Recommendations
Users should be aware that the model's responses are based on general fitness knowledge and are not specialized medical guidance.


## How to Get Started with the Model

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Soorya03/Llama-2-7b-chat-finetune", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Soorya03/Llama-2-7b-chat-finetune")

inputs = tokenizer("Give me a fitness tip", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


## Training Details

### Training Data

The model was fine-tuned on a fitness and exercise dataset (onurSakar/GYM-Exercise) to improve its domain knowledge in providing fitness-related responses.
<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
- **Method:** LoRA fine-tuning on top of Llama-2-7b-chat.
- **Hyperparameters:** Adjusted learning rate, FP16 precision for efficiency.
- **Compute:** Training was performed on Google Colab with a single GPU.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data
 
Sample fitness-related prompts were used for evaluation, but a formal benchmarking dataset was not utilized.

#### Metrics

Manual qualitative assessments showed the model’s suitability for fitness Q&A and general suggestions.

### Results

The model effectively generates coherent responses related to fitness, workouts, and exercise routines, with accurate language comprehension.


## Environmental Impact

### Compute Infrastructure

- **Hardware Type:** Google Colab (NVIDIA GPU)

## Model Architecture and Objective

This model is based on the Llama-2-7b-chat architecture, adapted to provide conversational responses within a specific fitness domain.

