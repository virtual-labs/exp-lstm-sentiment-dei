### Procedure

1. **Importing Libraries:**
   All necessary libraries for data handling, text preprocessing, model construction, training, and performance visualization are imported.

2. **Dataset Loading:**
   The IMDB movie review dataset is loaded by reading text files from the positive and negative review directories. Each review is assigned a sentiment label, where positive reviews are labelled as 1 and negative reviews as 0.

3. **Text Preprocessing:**
   The raw text reviews are cleaned to remove noise and irrelevant characters. This includes converting text to lowercase, removing punctuation marks, digits, special characters, and extra whitespaces.

4. **Data Splitting:**
   The cleaned dataset is divided into three subsets:
   - **Training set** for learning model parameters.
   - **Validation set** for hyperparameter tuning and model selection.
   - **Test set** for final performance evaluation.
   The split is performed in a balanced manner to maintain equal representation of positive and negative reviews.

5. **Text Tokenization:**
   The processed text is converted into numerical format using tokenization. A fixed vocabulary size is defined, and each unique word is mapped to a corresponding integer index. Words outside the vocabulary are replaced with an out-of-vocabulary token.

6. **Sequence Padding:**
   Since movie reviews vary in length, all tokenized sequences are padded or truncated to a fixed maximum sequence length. This ensures uniform input size for the neural network models.

7. **Embedding Layer Construction:**
   An embedding layer is added to both models to transform word indices into dense vector representations. This layer helps capture semantic relationships between words and improves model performance.

8. **Simple RNN Model Construction:**
   A Simple Recurrent Neural Network (RNN) model is built using an embedding layer followed by one or more recurrent layers. The final layer uses a sigmoid activation function to perform binary sentiment classification.

9. **LSTM Model Construction:**
   An LSTM model is constructed using an embedding layer followed by LSTM layers with gating mechanisms. Dropout and recurrent dropout are applied to reduce overfitting. A sigmoid-activated output layer is used for binary classification.

10. **Model Compilation:**
    Both RNN and LSTM models are compiled using:
    - **Binary Cross-Entropy** as the loss function.
    - **Adam optimizer** for efficient gradient-based optimization.
    - **Accuracy** as the evaluation metric.

11. **Model Training:**
    The models are trained on the training dataset for 150 epochs with 128 batch size. Learning rate scheduling is applied to reduce the learning rate when validation loss plateaus. Model checkpointing is used to save the best-performing model based on validation accuracy.

12. **Model Selection:**
    After training, the saved checkpoints corresponding to the highest validation accuracy are loaded. This ensures that the evaluation is performed using the best version of each model.

13. **Model Evaluation:**
    The trained RNN and LSTM models are evaluated on the unseen test dataset. Performance metrics include all accuracies, confusion matrix, and classification report consisting of precision, recall, and F1-score.

14. **Performance Visualization:**
    Learning curves for training and validation loss and accuracy are plotted to analyse model convergence and overfitting behaviour. Additionally, ROC curves and precision–recall curves are generated to assess classification performance more comprehensively.

15. **Performance Comparison:**
    Finally, the performance of the Simple RNN and LSTM models is compared based on accuracy, loss trends, ROC–AUC scores, and classification metrics to highlight the effectiveness of LSTM in handling long-term dependencies in sentiment analysis tasks, by analysing individual curves for both RNN and LSTM, as well as the combined curves.