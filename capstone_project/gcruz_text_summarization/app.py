from transformers import *
from summarizer import Summarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler
import streamlit as st

st.title('Extractive and Abstractive Text Summarization')
st.markdown('Using BERT and BART Transformer Models')

text = st.text_area('Please Input a Long Scientific Text')
abstract = st.text_area("Please Input Scientific Text Abstract")
conclusion = st.text_area("Please Input Scientific Text Conclusion")

pretrained_model = 'allenai/scibert_scivocab_uncased'

max_length = 750
min_length = 250

@st.cache(suppress_st_warning=True)
def get_summary(text, abstract, conclusion, pretrained_model):
    # Extractive Summarizer
    # Load model, model config and tokenizer via Transformers
    custom_config = AutoConfig.from_pretrained(pretrained_model)
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    custom_model = AutoModel.from_pretrained(pretrained_model, config=custom_config)

    # Create pretrained-model object
    extractive_model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

    # Abstractive Summarizer
    abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    optimal_num_sentences = extractive_model.calculate_optimal_k(text, k_max=10)
    extractive_summarized_text = "".join(extractive_model(text, num_sentences=optimal_num_sentences))
    
    text_list = [abstract, extractive_summarized_text, conclusion]
    joined_text = " ".join(text_list)

    abstractive_summary = abstractive_summarizer(joined_text, max_length=max_length, min_length=min_length, 
                                                do_sample=False)[-1]["summary_text"]
    st.write("Summary")
    st.success(abstractive_summary)

if st.button("Summarize"):
    get_summary(text, abstract, conclusion, pretrained_model)
