import streamlit as st
from transformers import pipeline
import wikipedia
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
st.title("Ask AI Tutor")
question = st.text_input("Ask your question:")
if question:
    try:
        topic = wikipedia.search(question)[0]
        context = wikipedia.summary(topic, sentences=3)
        result = qa_pipeline(question=question, context=context)
        st.write("Answer:", result['answer'])
    except Exception as e:
        st.error("Error: " + str(e))
