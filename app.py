import os
os.system("pip install gradio==3.0.18")
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import gradio as gr
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

def make_spans(text,results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(split_in_sentences(text),results_list))
    return facts_spans

##Fiscal Sentiment by Sentence
fin_model= pipeline("sentiment-analysis", model='FinanceInc/auditor_sentiment_finetuned', tokenizer='FinanceInc/auditor_sentiment_finetuned')
def fin_ext(text):
    results = fin_model(split_in_sentences(text))
    return make_spans(text,results)
    
##Forward Looking Statement
def fls(text):
    fls_model = pipeline("text-classification", model="FinanceInc/finbert_fls", tokenizer="FinanceInc/finbert_fls")
    results = fls_model(split_in_sentences(text))
    return make_spans(text,results) 

demo = gr.Blocks()

with demo:
    gr.Markdown("## Financial Analyst AI")
    gr.Markdown("This project applies AI trained by our financial analysts to analyze earning calls and other financial documents.")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Textbox(value="US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month. The department expects inflation to continue to rise.")
            with gr.Row():
                b5 = gr.Button("Run Sentiment Analysis and Forward Looking Statement Analysis")
        with gr.Column():
            with gr.Row():
                fin_spans = gr.HighlightedText()
            with gr.Row():
                fls_spans = gr.HighlightedText()
        b5.click(fin_ext, inputs=text, outputs=fin_spans)
        b5.click(fls, inputs=text, outputs=fls_spans)
    
demo.launch()