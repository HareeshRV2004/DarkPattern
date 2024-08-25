import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import LlamaForCausalLM, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration, pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import pdfplumber
import torch
import base64
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Directories and paths
UPLOAD_FOLDER = 'uploads'
EXTRACTED_TEXT_PATH = 'extracted_text.txt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load summarization model (LaMini-Flan-T5-248M)
sum_checkpoint = "LaMini-Flan-T5-248M"
offload_folder_248M = "offload_weights_248M"
os.makedirs(offload_folder_248M, exist_ok=True)

sum_tokenizer = T5Tokenizer.from_pretrained(sum_checkpoint)
sum_model = T5ForConditionalGeneration.from_pretrained(
    sum_checkpoint, 
    device_map='auto', 
    torch_dtype=torch.float32,
    offload_folder=offload_folder_248M
)

summary_pipeline = pipeline(
    "summarization",
    model=sum_model,
    tokenizer=sum_tokenizer,
    max_length=800,  
    min_length=100
)

# Load question-answering model (Lamini-T5-738M)
qa_checkpoint = "Lamini-T5-738M"
offload_folder_738M = "offload_weights_738M"
os.makedirs(offload_folder_738M, exist_ok=True)

qa_tokenizer = T5Tokenizer.from_pretrained(qa_checkpoint)
qa_model = T5ForConditionalGeneration.from_pretrained(
    qa_checkpoint, 
    device_map='auto', 
    torch_dtype=torch.float32,
    offload_folder=offload_folder_738M
)

qa_pipeline = pipeline(
    "text2text-generation",
    model=qa_model,
    tokenizer=qa_tokenizer
)

# Load dark pattern detection and sentiment analysis models
model_detect = pickle.load(open('model_detect.pkl', 'rb'))
model_presence = pickle.load(open('model_presence.pkl', 'rb'))
model = DistilBertForSequenceClassification.from_pretrained('./distilbert-sentiment-model')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-sentiment-model')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to scrape and extract text from a webpage
def scrape_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join([p.get_text() for p in soup.find_all('p')])

# Function to display PDF in HTML
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display



# Function to scrape reviews using Selenium
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def extract_reviews(driver, url, review_selector, author_selector, date_selector, rating_selector, text_selector, next_button_selector):
    driver.get(url)
    all_reviews = []

    while True:
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, review_selector)))
            reviews = driver.find_elements(By.CSS_SELECTOR, review_selector)

            for review in reviews:
                try:
                    review_text = review.find_element(By.CSS_SELECTOR, text_selector).text
                    review_author = review.find_element(By.CSS_SELECTOR, author_selector).text
                    review_date = review.find_element(By.CSS_SELECTOR, date_selector).text
                    review_rating = review.find_element(By.CSS_SELECTOR, rating_selector).get_attribute('innerText')

                    all_reviews.append({
                        'author': review_author,
                        'date': review_date,
                        'rating': review_rating,
                        'text': review_text
                    })
                except Exception as e:
                    print(f"Error extracting review: {e}")

            try:
                next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, next_button_selector)))
                if next_button:
                    next_button.click()
                    time.sleep(3)
                else:
                    break
            except Exception as e:
                print(f"Error navigating to next page: {e}")
                break

        except Exception as e:
            print(f"Error on page: {e}")
            break

    return all_reviews

def predict_sentiment(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.tolist()

# Web scraping function for dark pattern detection
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text("\n", strip=True)
        text_lines = text_content.split("\n")
        text_df = pd.DataFrame({'Index': range(1, len(text_lines) + 1), 'Text Content': text_lines})

        presence_predictions = model_presence.predict(text_lines)
        final_predictions = []

        for presence_prediction, text_line in zip(presence_predictions, text_lines):
            if presence_prediction == 'Dark':
                detect_prediction = model_detect.predict([text_line])[0]
                final_predictions.append(detect_prediction)
            else:
                final_predictions.append('Not Dark')
        text_df['Final Predictions'] = final_predictions

        return text_df

    except requests.exceptions.RequestException as err:
        return None

@app.route('/summarize_query', methods=['POST'])
def summarize_query():
    query = request.form['query']
    
    # Read the reviews from the CSV file
    reviews_df = pd.read_csv('scraped_reviews.csv')
    
    # Extract the 'text' column for the context
    context = " ".join(reviews_df['text'].tolist())  # Concatenate all review texts into a single string

    input_prompt = f"Context: {context}\n\nQuery: {query}\n\nSummary:"
    
    # Generate the summary using the Lamini-T5-738M model
    result = qa_pipeline(input_prompt, max_length=512, do_sample=True)[0]['generated_text']
    
    # Return the summary and query in a JSON response
    return jsonify({'query': query, 'summary': result})

flan_checkpoint = "LaMini-Flan-T5-248M"  # Replace with the actual local path
flan_offload_folder = "offload_weights_248M"
os.makedirs(flan_offload_folder, exist_ok=True)

# Load the local LMini-Flan-T5-248M model and tokenizer
flan_tokenizer = T5Tokenizer.from_pretrained(flan_checkpoint, cache_dir=flan_offload_folder)
flan_model = T5ForConditionalGeneration.from_pretrained(flan_checkpoint, cache_dir=flan_offload_folder)

# Function to load FAQ data
with open('faq.txt', 'r', encoding='utf-8') as file:
    faq_data = file.read()

def generate_faq_answer(question):
    input_text = f"Q: {question}\nA:"
    inputs = flan_tokenizer.encode(input_text, return_tensors='pt')
    outputs = flan_model.generate(inputs, max_length=200, num_return_sequences=1)
    answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("A:")[-1].strip()

@app.route('/ask_faq', methods=['POST'])
def ask_faq():
    data = request.get_json()
    question = data.get('question', '')
    answer = generate_faq_answer(question)
    return jsonify({'answer': answer})

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('dark.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/index')
def find():
    return render_template('firstpage.html')

@app.route('/review')
def review():
    return render_template('index1.html')

@app.route('/privacy_policy')
def privacy():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filepath.endswith('.pdf'):
        text = extract_text_from_pdf(filepath)
        pdf_view = display_pdf(filepath)
    else:
        with open(filepath, 'r') as f:
            text = f.read()
        pdf_view = f'<pre>{text}</pre>'

    with open(EXTRACTED_TEXT_PATH, 'w') as f:
        f.write(text)

    summary = summary_pipeline(text)[0]['summary_text']

    return render_template('summary.html', pdf_view=pdf_view, summary=summary, context=text, full_text=text)

@app.route('/scrape', methods=['POST'])
def scrape_url():
    url = request.form['url']
    text = scrape_text_from_url(url)

    with open(EXTRACTED_TEXT_PATH, 'w') as f:
        f.write(text)

    summary = summary_pipeline(text)[0]['summary_text']
    return render_template('summary.html', pdf_view=f'<pre>{text}</pre>', summary=summary, context=text, full_text=text)

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    context = request.form['context']
    pdf_view = request.form['pdf_view']
    summary = request.form['summary']
    full_text = request.form['full_text']

    input_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    result = qa_pipeline(input_prompt, max_length=512, do_sample=True)[0]['generated_text']

    return render_template('summary.html', pdf_view=pdf_view, summary=summary, context=context, full_text=full_text, question=question, answer=result)

@app.route('/result', methods=['POST'])
def result():
    url = request.form['url']
    text_df = scrape_website(url)

    if text_df is None:
        return render_template('error.html', message="An error occurred while fetching the website content.")

    dark_text_df = text_df[text_df['Final Predictions'] != 'Not Dark']

    if dark_text_df.empty:
        return render_template('error.html', message="No dark patterns were detected in the provided URL.")

    dark_patterns_count = dark_text_df['Final Predictions'].value_counts()
    labels = dark_patterns_count.index.tolist()
    values = dark_patterns_count.values.tolist()

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Dark Patterns')
    plt.axis('equal')
    pie_chart_path = 'static/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    not_dark_count_total = text_df['Final Predictions'].value_counts().get('Not Dark', 0)
    not_dark_count_filtered = dark_text_df['Final Predictions'].value_counts().get('Not Dark', 0)
    other_keywords_count_filtered = dark_text_df.shape[0] - not_dark_count_filtered
    other_keywords_count_total = text_df.shape[0] - not_dark_count_total

    not_dark_count_display = not_dark_count_total - not_dark_count_filtered
    other_keywords_count_display = other_keywords_count_total - other_keywords_count_filtered

    plt.figure(figsize=(10, 6))
    plt.bar(['Not Dark', 'Other Keywords'], [not_dark_count_display, other_keywords_count_display], color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Distribution of Dark Patterns in Text Content')
    bar_chart_path = 'static/bar_chart.png'
    plt.savefig(bar_chart_path)
    plt.close()

    return render_template('result.html', pie_chart_path=pie_chart_path, bar_chart_path=bar_chart_path, text_df=dark_text_df)

@app.route('/scrape_reviews', methods=['POST'])
def scrape_reviews():
    url = request.form['url']
    driver = setup_driver()
    
    amazon_reviews = extract_reviews(
        driver,
        url,
        'div[data-hook="review"]',
        'span.a-profile-name',
        'span[data-hook="review-date"]',
        'i[data-hook="review-star-rating"]',
        'span[data-hook="review-body"]',
        'li.a-last a'
    )
    
    driver.quit()
    
    reviews_df = pd.DataFrame(amazon_reviews)
    reviews_df.to_csv('scraped_reviews.csv', index=False)
    
    if 'text' not in reviews_df.columns:
        return jsonify({'error': 'Expected column "text" not found in the DataFrame.'}), 400
    
    texts = reviews_df['text'].tolist()
    predictions = predict_sentiment(texts)
    
    positive_count = predictions.count(1)
    negative_count = predictions.count(0)
    total_count = len(predictions)
    
    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100
    
    positive_reviews = [reviews_df.iloc[i]['text'] for i in range(len(predictions)) if predictions[i] == 1]
    negative_reviews = [reviews_df.iloc[i]['text'] for i in range(len(predictions)) if predictions[i] == 0]
    
    positive_summary = " ".join(positive_reviews[:3]) if positive_reviews else "No positive reviews found."
    negative_summary = " ".join(negative_reviews[:3]) if negative_reviews else "No negative reviews found."
    
    if positive_percentage > 50:
        recommendation = (
            "Based on the majority of positive feedback, this product seems to be well-received by users. "
            f"Key highlights include: {positive_summary}. However, be cautious of the following concerns: {negative_summary}."
        )
    else:
        recommendation = (
            "The product has received more negative feedback than positive. Users have pointed out the following issues: "
            f"{negative_summary}. Despite some positive aspects like: {positive_summary}, it might be worth considering other options."
        )
    
    result = {
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage,
        'positive_summary': positive_summary,
        'negative_summary': negative_summary,
        'recommendation': recommendation
    }
    
    return render_template('result1.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

