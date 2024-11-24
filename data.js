const glossaryTerms = [
    {
        "term": "A/B Testing",
        "definition": "A statistical method used to compare two versions of a model or system to determine which performs better against a defined metric."
    },
    {
        "term": "Accuracy",
        "definition": "A metric that measures the percentage of correct predictions made by a model out of all predictions."
    },
    {
        "term": "Activation Function",
        "definition": "A function applied to each node in a neural network to determine whether it should activate or not based on weighted input."
    },
    {
        "term": "Adversarial Attack",
        "definition": "A technique that attempts to fool machine learning models by creating deceptive input data."
    },
    {
        "term": "Artificial General Intelligence (AGI)",
        "definition": "A hypothetical type of AI that would have the ability to understand, learn, and apply knowledge across different domains at a human level."
    },
    {
        "term": "Artificial Intelligence",
        "definition": "A field of computer science focused on creating systems capable of performing tasks that require human intelligence."
    },
    {
        "term": "Attention Mechanism",
        "definition": "A technique in neural networks that allows the model to focus on specific parts of the input when producing output."
    },
    {
        "term": "Autoencoder",
        "definition": "A type of neural network that learns to compress data into a lower-dimensional representation and then reconstruct it."
    },
    {
        "term": "Backpropagation",
        "definition": "A training algorithm for neural networks that calculates gradients and adjusts weights to minimize errors."
    },
    {
        "term": "Bag of Words (BoW)",
        "definition": "A representation of text data where each document is represented as a set of words and their frequencies, disregarding grammar."
    },
    {
        "term": "Batch Normalization",
        "definition": "A technique used to standardize the inputs to a layer for each mini-batch to stabilize the learning process."
    },
    {
        "term": "Bias (in ML)",
        "definition": "A learnable parameter or systematic error in machine learning models that represents the offset from an activation function."
    },
    {
        "term": "Computer Vision",
        "definition": "A field of AI focused on enabling machines to interpret and make decisions based on visual data."
    },
    {
        "term": "Confusion Matrix",
        "definition": "A table used to evaluate classification model performance by showing true positives, false positives, true negatives, and false negatives."
    },
    {
        "term": "Convolutional Neural Network (CNN)",
        "definition": "A type of neural network commonly used in computer vision, specifically designed to process pixel data."
    },
    {
        "term": "Cross-Validation",
        "definition": "A resampling method used to evaluate model performance by partitioning data into training and validation sets multiple times."
    },
    {
        "term": "Data Augmentation",
        "definition": "A technique used to increase the diversity of training data by applying transformations like rotation, flipping, or cropping."
    },
    {
        "term": "Data Mining",
        "definition": "The process of discovering patterns, anomalies, and relationships in large datasets."
    },
    {
        "term": "Decision Tree",
        "definition": "A model used for classification and regression that splits data into branches to reach a decision based on certain conditions."
    },
    {
        "term": "Deep Learning",
        "definition": "A subset of machine learning that uses layered neural networks to analyze data with a high level of complexity."
    },
    {
        "term": "Dimensionality Reduction",
        "definition": "Techniques used to reduce the number of features in a dataset, making it easier to process and visualize."
    },
    {
        "term": "Dropout",
        "definition": "A regularization technique where randomly selected neurons are ignored during training to prevent overfitting."
    },
    {
        "term": "Embedding",
        "definition": "A representation of discrete variables as continuous vectors in a lower-dimensional space."
    },
    {
        "term": "Ensemble Learning",
        "definition": "A technique that combines multiple machine learning models to create a more robust and accurate model."
    },
    {
        "term": "Epoch",
        "definition": "One complete pass through the entire training dataset by a machine learning algorithm."
    },
    {
        "term": "Feature Engineering",
        "definition": "The process of creating new features or modifying existing ones to improve model performance."
    },
    {
        "term": "Feature Extraction",
        "definition": "The process of transforming raw data into numerical features that can be processed while maintaining the information in the original data set."
    },
    {
        "term": "Fine-tuning",
        "definition": "The process of taking a pre-trained model and further training it on a specific task or dataset."
    },
    {
        "term": "Generative Adversarial Network (GAN)",
        "definition": "A class of machine learning frameworks in which two networks compete, leading to improved data generation results."
    },
    {
        "term": "Gradient Descent",
        "definition": "An optimization algorithm used to minimize the loss function in machine learning and deep learning models."
    },
    {
        "term": "Hyperparameter",
        "definition": "Parameters in machine learning algorithms that are set before training and control the learning process, such as learning rate and batch size."
    },
    {
        "term": "Inference",
        "definition": "The process of using a trained model to make predictions on new, unseen data."
    },
    {
        "term": "K-Means Clustering",
        "definition": "An unsupervised learning algorithm that groups similar data points into clusters."
    },
    {
        "term": "K-Nearest Neighbors (KNN)",
        "definition": "A simple, instance-based learning algorithm that classifies data based on the closest neighbors in the feature space."
    },
    {
        "term": "Latent Dirichlet Allocation (LDA)",
        "definition": "A statistical model for topic modeling that classifies words in documents into different topics."
    },
    {
        "term": "Learning Rate",
        "definition": "A hyperparameter that controls how much to adjust the model in response to errors during training."
    },
    {
        "term": "Long Short-Term Memory (LSTM)",
        "definition": "A type of RNN architecture designed to handle long-term dependencies in sequential data."
    },
    {
        "term": "Machine Learning",
        "definition": "A subset of AI that uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."
    },
    {
        "term": "Natural Language Processing (NLP)",
        "definition": "A field of AI that enables computers to understand, interpret, and generate human language."
    },
    {
        "term": "Neural Network",
        "definition": "A series of algorithms that mimic the human brain to recognize patterns and relationships in data."
    },
    {
        "term": "One-Hot Encoding",
        "definition": "A process of converting categorical variables into a binary vector representation."
    },
    {
        "term": "Optimization",
        "definition": "The process of adjusting model parameters to minimize the loss function and improve performance."
    },
    {
        "term": "Overfitting",
        "definition": "A modeling error that occurs when a model learns the training data too well, including noise, making it less effective on new data."
    },
    {
        "term": "Principal Component Analysis (PCA)",
        "definition": "A dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space."
    },
    {
        "term": "RAG (Retrieval-Augmented Generation)",
        "definition": "A technique that enhances language models by combining them with a knowledge retrieval system to access external information during text generation, improving accuracy and factual consistency."
    },
    {
        "term": "Random Forest",
        "definition": "An ensemble learning technique that builds multiple decision trees and merges them to improve accuracy and reduce overfitting."
    },
    {
        "term": "Recurrent Neural Network (RNN)",
        "definition": "A type of neural network that processes sequential data by maintaining a memory of previous inputs."
    },
    {
        "term": "Regularization",
        "definition": "Techniques used to prevent overfitting by adding a penalty term to the loss function."
    },
    {
        "term": "Reinforcement Learning",
        "definition": "A type of machine learning where an agent learns by interacting with an environment to maximize some cumulative reward."
    },
    {
        "term": "Semi-Supervised Learning",
        "definition": "A machine learning method that combines a small amount of labeled data with a large amount of unlabeled data during training."
    },
    {
        "term": "Sentiment Analysis",
        "definition": "A technique used to determine the emotional tone or opinion expressed in text data."
    },
    {
        "term": "Supervised Learning",
        "definition": "A machine learning approach where models are trained on labeled data to make predictions."
    },
    {
        "term": "Support Vector Machine (SVM)",
        "definition": "A supervised machine learning model that separates data into classes by finding the optimal hyperplane."
    },
    {
        "term": "Tokenization",
        "definition": "The process of splitting text into smaller pieces, usually words or phrases, that can be used as inputs for NLP models."
    },
    {
        "term": "Transfer Learning",
        "definition": "A technique where a model developed for a particular task is reused as a starting point for a model on a second task."
    },
    {
        "term": "Transformer",
        "definition": "A neural network architecture that uses self-attention mechanisms to process sequential data, particularly effective in NLP tasks."
    },
    {
        "term": "Underfitting",
        "definition": "A situation where a machine learning model is too simple to capture the patterns in the data, resulting in poor performance."
    },
    {
        "term": "Unsupervised Learning",
        "definition": "A machine learning technique that analyzes and clusters unlabeled data to find hidden patterns or structures."
    },
    {
        "term": "Validation Set",
        "definition": "A subset of data used to evaluate model performance during training and tune hyperparameters."
    },
    {
        "term": "Word Embedding",
        "definition": "A learned representation of text where words with similar meaning have similar representations in vector space."
    }
];
