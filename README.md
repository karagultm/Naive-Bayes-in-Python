# Naive-Bayes-in-Python

Project Overview
This project involves implementing a Naive Bayes classifier from scratch to classify instances in the play_tennis.csv dataset. The goal is to predict whether a game of tennis will be played based on weather conditions such as Outlook, Temperature, Humidity, and Wind.

Through this project, you will:

Understand how the Naive Bayes algorithm works.
Gain experience in calculating probabilities and implementing classification logic manually.
Evaluate model performance using metrics like accuracy and a confusion matrix.
Dataset
The dataset used for this project is play_tennis.csv. It contains:

Features:
Outlook: (Sunny, Overcast, Rain)
Temperature: (Hot, Mild, Cool)
Humidity: (High, Normal)
Wind: (Weak, Strong)
Target Variable: PlayTennis (Yes/No)
The dataset is provided in a semicolon-delimited CSV format.

How to Run the Code
Requirements:

Python 3.6 or above
Libraries:
bash
Kodu kopyala
pip install numpy pandas
Files:

play_tennis.csv: The dataset file.
naive_bayes_classifier.py: The main Python script containing the implementation.
likelihoods.json: Automatically generated file for storing likelihood probabilities.
naive_bayes_log.txt: Log file generated during execution.
Steps to Execute:

Place play_tennis.csv in the same directory as the script.
Run the script using the command:
bash
Kodu kopyala
python naive_bayes_classifier.py
Check the console for accuracy results.
Outputs:

Confusion Matrix: Shows the classifier's performance.
Accuracy: Displays the percentage of correct predictions.
Logs: Detailed execution logs are stored in naive_bayes_log.txt.
Project Structure
Introduction: Explains the Naive Bayes algorithm and its use case in this project.
Implementation: Demonstrates step-by-step calculations, including priors, likelihoods, and posterior probabilities.
Evaluation: Includes accuracy calculations and confusion matrix analysis.
Discussion: Highlights strengths, challenges, and limitations.
Conclusion: Summarizes insights gained from the project.
Features of the Naive Bayes Classifier
Custom Implementation: Avoids external libraries like scikit-learn to understand the algorithm in detail.
Handles Zero Probabilities: Implements Laplace smoothing to address missing feature values.
Performance Metrics: Includes accuracy and confusion matrix for evaluation.
Limitations and Future Work
Assumption of Feature Independence: This classifier assumes features are independent, which might not be true for all datasets.
Small Dataset: The small size of the dataset may limit the model's generalizability.
Future Improvements: Extend the implementation to handle continuous features or experiment with larger datasets.
References
Bayes’ Theorem and Naive Bayes Algorithm: Wikipedia
Python Libraries: pandas and JSON
Dataset: Custom dataset play_tennis.csv.
https://medium.com/@rangavamsi5/naïve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9