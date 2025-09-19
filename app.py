from transformers import pipeline, MBart50TokenizerFast, AutoTokenizer, AutoModelForTokenClassification
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("news.html", text="")

@app.route("/summarize", methods=["POST"])
def summarize():
    model_name="sshleifer/distilbart-cnn-6-6"
    summarizer = pipeline("summarization", model=model_name)
    text = request.form.get("text")
    s = summarizer(text, max_length=350, min_length=50, do_sample=False)
    summary = s[0]['summary_text']
    return render_template("summarize.html", summary=summary)

@app.route("/Translate", methods=["POST"])
def translate():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    translator = pipeline("translation", model=model_name, tokenizer=tokenizer)

    text = request.form.get("text")
    translated_text = translator(text, src_lang="en_XX", tgt_lang="hi_IN")
    translated = translated_text[0]['translation_text']

    return render_template("translate.html", translated=translated)

@app.route("/ner", methods=["POST"])
def ner():
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    text = request.form.get("text")
    ner_results = nlp(text)
    return render_template("ner.html", ner_results=ner_results)
  
if __name__ == "__main__":
    app.run(debug=True)