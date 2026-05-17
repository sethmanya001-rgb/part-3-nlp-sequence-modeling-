# part-3-nlp-sequence-modeling-
Dataset:https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

Tasks
Copy everything below and paste into README.md file:
Part 3: NLP and Sequence Modeling Mini Project
Domain: Customer Support - Sentiment Classification
Overview
This project builds an NLP pipeline to classify customer support messages into 3 sentiment categories. It compares traditional text vectorization methods with a deep learning LSTM sequence model.
Dataset
File: customer_support_text_classification.csv
Total records: 1500
Target variable: sentiment_label
Classes are positive, neutral, and negative
Average message length: 12.7 words
Dataset is nearly balanced across all 3 classes
Tasks Completed
Task 1: Dataset explored with record count, class distribution, sample texts, and average text length
Task 2: Text cleaned with lowercasing, special character removal, tokenization, and stopword removal
Task 3: Text vectorized using TF-IDF and Bag of Words with explanation of why text must become numbers
Task 4: Baseline models built using Logistic Regression with TF-IDF and Naive Bayes with Bag of Words
Task 5: LSTM sequence model built with Embedding layer, LSTM layer, Dense layers, and Softmax output
Task 6: Written reflection on RNNs, LSTMs, Attention mechanism, and Transformers
Models Built
Baseline Model 1 is Logistic Regression with TF-IDF vectorization
Baseline Model 2 is Naive Bayes with Bag of Words vectorization
Deep Learning Model is LSTM with Embedding layer and sequence padding
LSTM Architecture
Embedding layer with input dimension 5000 and output dimension 64
LSTM layer with 64 units
Dropout 0.3
Dense layer with 32 neurons and ReLU
Dropout 0.2
Output Dense layer with 3 neurons and Softmax
Loss function: Sparse Categorical Crossentropy
Optimizer: Adam
Max sequence length: 50 words
Vocabulary size: 5000 words
Key Concepts Explained
RNNs struggle with long sequences because information from early words fades away by the time the model reaches later words. This is called the vanishing gradient problem. LSTMs solve this with forget gates, input gates, and output gates that control what information to remember and what to discard. Attention allows the model to look at all words at the same time and focus on the most important ones regardless of their position. Transformers use only attention with no RNN at all which makes them much faster and better at understanding long documents. Famous transformer models include BERT, GPT-4, and Claude which are the foundation of modern Generative AI.
How to Run
Install dependencies using: pip install -r requirements.txt
Open notebook using: jupyter notebook notebook.ipynb
Run all cells from top to bottom
Requirements
tensorflow 2.13 or higher
numpy 1.24 or higher
pandas 2.0 or higher
scikit-learn 1.3 or higher
matplotlib 3.7 or higher
nltk 3.8 or higher


