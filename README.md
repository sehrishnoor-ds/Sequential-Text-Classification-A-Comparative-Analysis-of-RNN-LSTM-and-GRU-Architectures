# Sequential-Text-Classification-A-Comparative-Analysis-of-RNN-LSTM-and-GRU-Architectures

## 📌 Project Overview
This repository contains a deep learning project that implements and analyzes different Recurrent Neural Network (RNN) architectures for sequence modeling. The primary objective is to classify text data (sentiment analysis) using the **IMDB Movie Reviews Dataset** and to evaluate the performance, computational efficiency, and theoretical trade-offs between a Vanilla RNN, an LSTM, and a GRU.

This project was completed as part of an assignment to demonstrate the practical handling of vanishing gradients in sequential data using gating mechanisms.

## 🚀 Key Features & Tasks

* ** Data Preparation**
  * Loaded the built-in IMDB dataset using `tf.keras.datasets`.
  * Preprocessed and padded text sequences to a fixed length (200 words) using `pad_sequences`.
  * Visualized sequence structure by decoding integer maps back to English text.
* ** Model Implementation**
  * Built three distinct models using the `tf.keras.Sequential` API:
    1. **Simple RNN** (`tf.keras.layers.SimpleRNN`)
    2. **LSTM** (`tf.keras.layers.LSTM`)
    3. **GRU** (`tf.keras.layers.GRU`)
  * Utilized word embeddings (`tf.keras.layers.Embedding`) for text representation.
* ** Performance Comparison**
  * Evaluated and recorded Test Accuracy, Test Loss, and Training Time for all three architectures.
  * Generated visualizations for Validation Accuracy over epochs.
  * Compared sample predictions side-by-side to observe architectural strengths on complex reviews.
* ** Theoretical Reflection**
  * Analyzed real-world results against deep learning theory.
  * Demonstrated how LSTMs and GRUs successfully overcome the **vanishing gradient problem** through their respective memory gating mechanisms (Forget/Input/Output gates for LSTM; Reset/Update gates for GRU).
  * Highlighted Keras-specific hardware optimization quirks (e.g., CuDNN acceleration making LSTMs/GRUs compute faster than standard RNNs).

## 📊 Empirical Results

| Model | Test Accuracy | Test Loss | Train Time (s) |
| :--- | :--- | :--- | :--- |
| **Simple RNN** | 84.97% | 0.3678 | 14.53 |
| **LSTM** | **86.59%** | **0.3236** | 9.75 |
| **GRU** | 85.95% | 0.3345 | **7.17** |

* **Accuracy Winner:** **LSTM** successfully captured the most complex long-term dependencies in the text.
* **Efficiency Winner:** **GRU** provided near-LSTM accuracy while training the fastest due to its simplified two-gate architecture.
* **Unexpected Finding:** Despite having fewer parameters, the **Simple RNN** took the longest to train because it lacks the low-level hardware optimizations (like CuDNN) inherently utilized by TensorFlow's LSTM and GRU layers.

## 🛠️ Technologies Used
* Python 3
* TensorFlow / Keras (Deep Learning Framework)
* NumPy (Data Manipulation)
* Matplotlib (Data Visualization)

## 💻 How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed: `pip install tensorflow numpy matplotlib`.
3. Open the Jupyter Notebook (`.ipynb` file).
4. Run the cells sequentially. The dataset will be downloaded automatically via Keras upon the first run.
*(Note: This code is highly optimized to run smoothly within a Kaggle Notebook environment).*
