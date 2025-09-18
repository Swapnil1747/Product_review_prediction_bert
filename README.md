# E-commerce Customer Review Analysis System

A state-of-the-art AI-powered web application for analyzing customer feedback from e-commerce product reviews to predict potential churn risk. Built with BERT (Bidirectional Encoder Representations from Transformers) and deployed using Gradio for an intuitive user interface.

## 🚀 Features

- **Advanced NLP Analysis**: Fine-tuned BERT model for accurate sentiment classification
- **Real-time Prediction**: Instant analysis of customer reviews with confidence scores
- **Strategic Recommendations**: Actionable insights based on prediction results
- **Security-First Design**: Rate limiting, input validation, and sanitization
- **Modern UI**: Dark theme with gradient styling and responsive design
- **Enterprise-Grade**: Scalable architecture with error handling

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Gradio 4.0+

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd REview-Prediction-from-E-commerce-Product-using-Bert-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model files are in the `./model` directory.

## 🎯 Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to the provided URL (typically `http://localhost:7860` or the shared link).

3. Enter a customer review in the text box and click "Analyze Customer Feedback".

4. View the results:
   - **Review Risk Assessment**: Positive or Negative classification
   - **Prediction Confidence**: Confidence percentage
   - **Analysis Status**: Success or error messages
   - **Strategic Recommendation**: Suggested actions based on the analysis

## 🤖 Model Details

- **Base Model**: BERT (bert-base-uncased)
- **Task**: Binary sequence classification (Positive vs Negative sentiment)
- **Training Data**: E-commerce product reviews dataset
- **Fine-tuning**: 2 epochs with early stopping
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Preprocessing**: Text cleaning, tokenization (max_length=128)

The model maps:
- **Positive (0)**: Satisfied customer, low churn risk
- **Negative (1)**: Dissatisfied customer, high churn risk

## 🔒 Security Features

- **Rate Limiting**: 10 requests per minute per IP
- **Input Validation**: Length checks (10-1000 characters)
- **Sanitization**: XSS prevention and HTML tag removal
- **Error Handling**: Graceful failure with user-friendly messages

## 📊 Training the Model

To retrain or fine-tune the model:

1. Open `bert_model_finetune.ipynb` in Jupyter Notebook
2. Ensure `product_review.csv` is in the project root
3. Run the cells to preprocess data, train the model, and save it
4. Update the model path in `app.py` if necessary

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Swapnil Mishra**

---

*Enterprise-Grade E-commerce Customer Analytics Platform | Powered by Advanced AI & Machine Learning*
