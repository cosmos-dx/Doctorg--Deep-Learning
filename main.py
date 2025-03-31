import sys
sys.modules["sqlite3"] = __import__("pysqlite3")


import streamlit as st
import spacy
import numpy as np
import pandas as pd
import tensorflow.lite as tflite
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import os
from crewai import Crew, Agent, LLM, Task
import google.generativeai as genai


model_llm = genai.GenerativeModel("models/gemini-2.0-flash-exp")
st.set_page_config(page_title="Disease Prediction AI", page_icon="🩺", layout="wide")


st.sidebar.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #353935;
            padding: 20px;
        }
        
        [data-testid="stSidebar"] h1 {
            font-size: 2rem;
            color: white;
            text-align: center;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 1.8rem;
            color: white;
            text-align: center;
        }

        [data-testid="stSidebar"] .stMarkdown {
            text-align: center;
        }

        [data-testid="stSidebar"] a {
            display: block;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 5px 0;
            color: white !important;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.2rem;
        }

        [data-testid="stSidebar"] a:hover {
            opacity: 0.8;
        }

        .stButton>button {
            width: 100%;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<h1>🩺 DoctorG</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>📌 Connect with Me</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <a href="https://github.com/cosmos-dx" style="background:#000;color:white;">GitHub</a>
    <a href="https://www.linkedin.com/in/abhishek-gupta-a1a44a203/" style="background:#0077B5;color:white;">LinkedIn</a>
    <a href="https://x.com/cosmos_dx" style="background:#000;color:white;">X (Twitter)</a>
    <a href="https://www.youtube.com/@gloryofcode" style="background:#FF0000;color:white;">YouTube</a>
    <a href="https://www.instagram.com/abhishek.gupta0118" style="background:#E4405F;color:white;">Instagram</a>
    <a href="https://www.facebook.com/profile.php?id=100002567964281" style="background:#1877F2;color:white;">Facebook</a>
    <a href="https://www.cosmosdx.me/" style="background:#FF5733;color:white;">Portfolio</a>
    <a href="https://github.com/cosmos-dx/Doctorg--Deep-Learning" style="background:#009688;color:white;">Dataset</a>
    """,
    unsafe_allow_html=True,
)

try:
    nlp = spacy.load("./en_core_web_sm/en_core_web_sm-3.8.0")
except OSError:
    st.write("Downloading en_core_web_sm language model for SpaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# model_path = "en_core_web_sm"

# # Check if the model directory exists locally
# if os.path.exists(model_path):
#     nlp = spacy.load(model_path)
# else:
#     st.write(f"Model path {model_path} does not exist. Please ensure the model is present.")


df = pd.read_csv("doctorg_processed.csv")
label_encoder = LabelEncoder()
df["disease_encoded"] = label_encoder.fit_transform(df["name"])
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

interpreter = tflite.Interpreter(model_path="doctorg_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_my = LLM(model="gemini/gemini-2.0-flash-exp", temperature=0.7)

agent = Agent(
    role="Disease Information Specialist",
    goal="Provide users with informative guidance regarding their symptoms, suggest remedies, and advise on next steps.",
    backstory="A compassionate AI specializing in providing initial guidance on health concerns.",
    llm=llm_my,
)


def extract_symptoms(text):
    """
    Extracts symptoms from the input text using NLP and LLM for accuracy.
    """
    prompt = f"""Please extract the symptoms mentioned in the following text and return them as a comma-separated list:
    \"{text}\" Symptoms:"""
    
    response = model_llm.generate_content(prompt)
    symptom_string = response.text.strip()
    extracted_symptoms = [s.strip() for s in symptom_string.split(',')]

    processed_symptoms = []
    for symptom in extracted_symptoms:
        doc = nlp(symptom.lower())
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        processed_symptoms.append(" ".join(filtered_tokens).strip())

    final_symptoms = [s for s in processed_symptoms if s]

    if len(final_symptoms) < 5:
        st.warning("Please provide at least 5 symptoms for better accuracy.")
        if not st.button("I understand, let me add more symptoms"):
            st.stop()

    return final_symptoms

def predict_disease(user_symptoms):
    user_embedding = sentence_model.encode(" ".join(user_symptoms)).reshape(1, -1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], user_embedding)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    top_5_diseases = [label_encoder.classes_[i] for i in top_5_indices]
    top_5_probs = [predictions[i] for i in top_5_indices]

    return list(zip(top_5_diseases, top_5_probs))

def get_disease_description(disease_name):
    disease_row = df[df['name'] == disease_name]
    return disease_row['description'].iloc[0] if not disease_row.empty else "No description available."

def create_disease_info_task(disease_description, top_disease):
    return Task(
        description=f"""Based on your mentioned symptoms, a potential condition is '{top_disease}'.
        Here is a brief description:

        {disease_description}

        Please provide guidance including reassurance, general remedies, information, and next steps.""",
        agent=agent,
        expected_output="A helpful and informative response about the potential condition."
    )

st.title("🩺 DoctorG: AI-Powered Disease Prediction")
st.write("Enter your symptoms in natural language and get AI-powered guidance!")

user_input = st.text_area("Describe your symptoms:", "")

if st.button("Check Probable Disease"):
    if user_input.strip():
        extracted_symptoms = extract_symptoms(user_input)
        
        if extracted_symptoms:
            predicted_diseases = predict_disease(extracted_symptoms)
            
            if predicted_diseases:
                st.subheader("Predicted Diseases")
                for disease, probability in predicted_diseases:
                    st.write(f"- **{disease}** with a probability of {probability * 100:.2f}%")
                top_disease = predicted_diseases[0][0]
                description = get_disease_description(top_disease)
                
                st.subheader("Detailed Guidance")
                task = create_disease_info_task(description, top_disease)
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                st.markdown(result)
            else:
                st.error("Could not predict any disease based on the provided symptoms.")
    else:
        st.warning("Please enter symptoms before predicting.")

st.markdown("---")
st.markdown("<h3 style='text-align:center;'>Abhishek Gupta</h3>", unsafe_allow_html=True)
