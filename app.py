import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
from datetime import datetime, timedelta
from collections import defaultdict
import html
import os

# Security configurations
MAX_REVIEW_LENGTH = 1000  # Maximum characters in review
MIN_REVIEW_LENGTH = 10    # Minimum characters in review
RATE_LIMIT = 10           # Maximum requests per minute
RATE_WINDOW = 60          # Time window in seconds

# Rate limiting storage
request_history = defaultdict(list)

def is_rate_limited(ip):
    """Check if the IP has exceeded the rate limit"""
    now = datetime.now()
    # Remove old requests outside the window
    request_history[ip] = [t for t in request_history[ip] if now - t < timedelta(seconds=RATE_WINDOW)]
    # Check if rate limit is exceeded
    return len(request_history[ip]) >= RATE_LIMIT

def sanitize_input(text):
    """Sanitize input text to prevent XSS and other attacks"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Escape HTML characters
    text = html.escape(text)
    # Remove any remaining special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def validate_input(text):
    """Validate input text"""
    if not text or not isinstance(text, str):
        return False, "Invalid input format"
    
    text = text.strip()
    if len(text) < MIN_REVIEW_LENGTH:
        return False, f"Review must be at least {MIN_REVIEW_LENGTH} characters long"
    if len(text) > MAX_REVIEW_LENGTH:
        return False, f"Review must not exceed {MAX_REVIEW_LENGTH} characters"
    
    return True, text

# Load model and tokenizer
MODEL_PATH = os.getenv("MODEL_PATH", "./model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {0: "Positive", 1: "Negative"}

def get_suggestion(label, confidence):
    if label == "Churn":
        if confidence >= 80:
            return "⚠️ High Risk: Strong indication of customer dissatisfaction. Immediate action recommended."
        elif confidence >= 60:
            return "⚠️ Moderate Risk: Customer shows signs of dissatisfaction. Proactive engagement advised."
        else:
            return "⚠️ Low Risk: Some concerns detected. Consider reaching out to understand customer needs better."
    else:  # Not Churn
        if confidence >= 80:
            return "✅ High Confidence: Customer shows strong satisfaction. Continue current engagement strategy."
        elif confidence >= 60:
            return "✅ Moderate Confidence: Customer appears satisfied. Regular check-ins recommended."
        else:
            return "✅ Low Confidence: Customer seems satisfied but monitor for any changes in behavior."

def predict_churn(review_text, request: gr.Request):
    """Predict churn with security measures"""
    # Rate limiting check
    if is_rate_limited(request.client.host):
        return "", 0.0, "⚠️ Rate limit exceeded. Please try again later.", ""

    # Input validation
    is_valid, validation_result = validate_input(review_text)
    if not is_valid:
        return "", 0.0, f"⚠️ {validation_result}", ""

    # Sanitize input
    review_text = sanitize_input(review_text)

    try:
        # Record the request
        request_history[request.client.host].append(datetime.now())

        inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)
        label = id2label.get(predicted_id.item(), "Unknown")

        confidence_percent = round(confidence.item() * 100, 2)
        message = "✅ Analysis completed successfully."
        suggestion = get_suggestion(label, confidence_percent)

        return label, confidence_percent, message, suggestion

    except Exception as e:
        return "", 0.0, f"❌ Analysis error: {str(e)}", ""

css = """
body {
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #f0f0f0;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

#header {
    text-align: center;
    margin: 2rem 0;
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#description {
    text-align: center;
    font-size: 1.1rem;
    max-width: 800px;
    margin: 0 auto 2rem;
    line-height: 1.6;
    color: #b8c6db;
}

.gradio-container {
    max-width: 1000px;
    margin: 2rem auto;
    background: rgba(255, 255, 255, 0.05);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.input-box {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.input-box:focus {
    border-color: #4facfe !important;
    box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.2) !important;
}

.output-box {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    min-height: 60px !important;
}

#predict_btn {
    background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3) !important;
}

#predict_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4) !important;
}

#predict_btn:active {
    transform: translateY(0) !important;
}

#about {
    text-align: center;
    font-size: 0.9rem;
    color: #8a9bb8;
    padding-top: 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Label styling */
label {
    color: #b8c6db !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
    gr.Markdown("""
        <h1 id="header">E-commerce customer Review Analysis System</h1>
        <p id="description">
        Leverage advanced AI to analyze customer feedback and predict potential Negative Risk. 
        Our state-of-the-art BERT model processes product reviews to identify early warning signs 
        of customer dissatisfaction, enabling proactive retention strategies and data-driven decision making.
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            review_input = gr.Textbox(
                label="Customer Review Analysis",
                placeholder="Enter the customer review for analysis...",
                lines=6,
                max_lines=12,
                elem_id="input_textbox",
                elem_classes="input-box",
                interactive=True,
                max_length=MAX_REVIEW_LENGTH
            )
            predict_btn = gr.Button("Analyze Customer Feedback", elem_id="predict_btn")

        with gr.Column(scale=1):
            output_label = gr.Textbox(
                label="Review Risk Assessment",
                interactive=False,
                elem_classes="output-box"
            )
            confidence_text = gr.Textbox(
                label="Prediction Confidence",
                interactive=False,
                elem_classes="output-box"
            )
            status_message = gr.Textbox(
                label="Analysis Status",
                interactive=False,
                elem_classes="output-box"
            )
            suggestion_box = gr.Textbox(
                label="Strategic Recommendation",
                interactive=False,
                elem_classes="output-box"
            )

    predict_btn.click(
        fn=predict_churn,
        inputs=[review_input],
        outputs=[output_label, confidence_text, status_message, suggestion_box],
        show_progress=True
    )

    gr.Markdown("""
        <div id="about">
            Enterprise-Grade E-commerce Customer Analytics Platform | Powered by Advanced AI & Machine Learning<br>
            <span style="color: #8a9bb8; font-size: 0.9rem;">Developed by <b>Swapnil Mishra</b></span>
        </div>
    """)

# Launch the Gradio app on all interfaces, port 7860, with public sharing enabled
demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
