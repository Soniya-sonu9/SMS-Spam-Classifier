# 📧 Email Spam Detector

A machine learning-based application to classify emails or messages as **spam** or **not spam** using Natural Language Processing (NLP) and classification algorithms.

## 🚀 Project Overview

This project demonstrates how to build a spam detector that can distinguish between legitimate and spam messages. It involves preprocessing text data, training a classification model, and deploying it via a simple Flask application.

## 📂 Project Structure

```
├── Email_spam_detector (1).ipynb        # Jupyter Notebook for exploration & modeling
├── train_model.py                       # Python script to train and save the model
├── spam (1).csv                         # Dataset containing email/message data
├── model.pkl                            # Trained classification model
├── vectorizer.pkl                       # TF-IDF vectorizer
├── app.py                               # Flask app for deployment
├── Email-Spam-Detection/                # App configuration files
```

## 🧠 Technologies & Concepts Used

- Python 🐍
- Natural Language Processing (NLP)
- Scikit-learn (TF-IDF, Naive Bayes or similar classifier)
- Pandas & NumPy
- Flask (for deployment)
- Pickle (for model serialization)

## 📊 Dataset

The dataset (`spam (1).csv`) consists of labeled messages with two main columns:
- `label`: 'spam' or 'ham'
- `text`: the content of the message

## ⚙️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (optional)**
   ```bash
   python train_model.py
   ```

3. **Run the web app**
   ```bash
   python app.py
   ```

4. Open your browser at `http://localhost:5000` to test the spam detector.

## 🧪 Model Output

The model takes message input from a web form and returns whether it is **Spam** or **Not Spam**, based on trained machine learning logic.

## 🔬 Future Improvements

- Improve accuracy with deep learning models (e.g., LSTM)
- Use larger, more diverse datasets
- Integrate database for message history
- Add REST API support

## 📄 License

This project is for educational purposes only.
