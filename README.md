# Word2Vec Wizardry: Discovering Similar Words with NLP

This project focuses on training a Word2Vec model using a [dataset](https://huggingface.co/datasets/sentence-transformers/all-nli) from [HuggingFace](https://huggingface.co/) to predict similar words. The model leverages linguistic preprocessing techniques to clean and prepare text data before training, ensuring accurate and meaningful word embeddings for NLP applications.

## Functionalities

* **Leverages Hugging Face Datasets:** Employs the `datasets` library from Hugging Face to load a pre-existing dataset (sentence-transformers/all-nli) containing sentence pairs.
* **Data Preprocessing:** Performs text cleaning steps like tokenization, stop word removal, and lemmatization using NLTK libraries to prepare the training data for the Word2Vec model.
* **Word2Vec Model Training:** Trains a Word2Vec model using the `gensim` library with hyperparameters tuned for similar word prediction.
* **Similar Word Prediction:** Defines a function `get_similar_word` that takes a word as input and retrieves the top N most similar words from the trained model's vocabulary, along with their similarity scores.


### Table of Contents:

1. [Installing Libraries](#installing-libraries)
2. [Importing Libraries](#importing-libraries)
3. [Loading Dataset](#loading-dataset)
4. [Preparing Dataset](#preparing-dataset)
5. [Preprocessing Data](#preprocessing-data)
6. [Training Word2Vec Model](#training-word2vec-model)
7. [Predicting Similar Words](#predicting-similar-words)
8. [Conclusion](#conclusion)

## Installing Libraries

Installs the required libraries (`datasets`, `nltk`) using pip (`!pip install datasets`).

## Importing Libraries

Imports necessary libraries for data manipulation (`nltk`), text pre-processing (`nltk`), dataset loading (`datasets`), and Word2Vec model building (`gensim`).

## Loading Dataset

Loads a pre-defined sentence similarity dataset from Hugging Face using `load_dataset()`.

## Preparing Dataset

   - Extracts sentence pairs from the dataset.
   - Combines sentences from training, testing, and validation sets to create a comprehensive training dataset.
   - Removes duplicates to ensure unique training examples.

## Preprocessing Data

Defines a function `linguistic_preprocessing` that performs the following steps on each sentence:
   - Tokenization (breaking down into words).
   - Stop word removal (eliminating common words like "the", "a", etc.).
   - Lemmatization (reducing words to their base form).

## Training Word2Vec Model

Creates and trains a Word2Vec model using `gensim.models.Word2Vec` with specified hyperparameters:
   - Vector size: dimensionality of the word embeddings.
   - Window size: context window for considering surrounding words.
   - Minimum count: minimum frequency of a word to be included.
   - Training epochs: number of times to iterate through the training data.

## Predicting Similar Words

Defines a function `get_similar_word()` that:
   - Takes a word as input.
   - Retrieves the top 5 most similar words from the model's vocabulary based on their cosine similarity scores.
   - Prints the similar words with their corresponding scores.

## Conclusion

This project demonstrates the effective use of Word2Vec for predicting similar words. By leveraging HuggingFace's dataset and implementing robust preprocessing techniques, the model allows users to explore word relationships and find semantically similar words within the trained model's vocabulary. The model achieves accurate and meaningful word embeddings, enhancing text analysis and NLP applications.
