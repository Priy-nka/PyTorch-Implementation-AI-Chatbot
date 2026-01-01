# PyTorch-Implementation-AI-Chatbot

A deep learning-powered chatbot built with PyTorch and NLTK. This assistant uses a Feed-Forward Neural Network to classify user intents based on natural language patterns and provides relevant responses from a customizable JSON configuration.

## 🧠 Project Architecture

The chatbot utilizes a multi-layer perceptron (MLP) architecture:

Input Layer: Bag-of-Words (BoW) vector representing lemmatized tokens.

Hidden Layers: Two fully connected layers with 128 and 64 units, utilizing ReLU activation and Dropout (0.5) to prevent overfitting.

Output Layer: Linear layer corresponding to the number of intent tags.

## 📁 Key Files

main.py: The core logic containing the ChatbotAssistant and ChatbotModel classes.

intents.json: The configuration file containing patterns, tags, and responses.

chatbot_model.pth: The saved state dictionary of the trained PyTorch model.

dimensions.json: Stores the input/output shapes for consistent model loading.

## 🛠️ Features

Natural Language Processing: Uses NLTK for tokenization and lemmatization to handle word variations.

Dynamic Intent Mapping: Support for function_mappings, allowing the chatbot to trigger specific Python functions based on detected intents.

Customizable Training: Easily adjust hyperparameters like batch size, learning rate, and epochs within the train_model method.
