# Training a Llama2-Powered Chatbot to Interact with Research Papers
![Chatbot Diagram](https://github.com/anair123/Llama2-Powered-QA-Chatbot-For-Research-Papers/assets/47230033/5ea10939-b3a3-48c7-819e-9af6db879dbc)
![Chatbot Diagram](https://ibb.co/HGf574X)


## Introduction
The goal of this project is to build a closed-source chatbot on a CPU using the quantized Llama2 model (7B parameters).

The resulting application will be evaluated based on it's ability as a tool of convenience for retrieving information from research papers. More specifically, it will evaluated by the quality of it's responses, the run time, and the memory expenditure. 

## Installation Instructions

1. Clone this repository using the command:  
```git clone https://github.com/anair123/Llama2-Powered-QA-Chatbot-For-Research-Papers.git```

2. Download a quantized Llama2 model (pick any one) from the following link:     https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

3. Store the model in the "models" directory

4. Create a virtual environment and enter it  
```python -m venv <name_of_venv>```  
```venv/Scripts/Activate```

5. Install the dependencies with the command:  
```pip install -r requirements.txt```

6. Add all the pdf documents you want to interact with in the "data" folder.

7. Run the Streamlit web app with the command:  
```streamlit run app.py```



## Author
Rania Hossam  
LinkedIn: www.linkedin.com/in/rania-hossam55
Medium: [](https://medium.com/@raniahossam)https://medium.com/@raniahossam


