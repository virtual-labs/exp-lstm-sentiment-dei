### Procedure

1. **Importing Libraries:**
   All required Python libraries for numerical computation, text preprocessing, deep learning model construction, training, evaluation, and visualization are imported. These include NumPy, Matplotlib, Seaborn, TensorFlow/Keras, and Scikit-learn.


2. **Dataset Loading:**
   The IMDB Movie Reviews dataset is loaded by reading text files from the predefined positive and negative review directories. To ensure manageable training time and controlled experimentation, a balanced subset of the dataset is used. Each review is assigned a sentiment label, where positive reviews are labelled as 1 and negative reviews as 0.


3. **Text Preprocessing:**
   The raw movie reviews are cleaned using regular expression-based preprocessing techniques. This includes:
   - Converting all text to lowercase.
   - Removing punctuation marks, digits, and special characters.
   - Eliminating extra whitespaces.

   This step helps reduce noise and improves the quality of textual features learned by the models. 

4. **Data Splitting:**
   The cleaned dataset is divided into three subsets:
   - **Training set** for learning model parameters.
   - **Validation set** for hyperparameter tuning and model selection.
   - **Test set** for final performance evaluation.

   A stratified splitting strategy is used to ensure equal representation of positive and negative reviews across all subsets.

5. **Text Tokenization:**
   The preprocessed text data is converted into numerical format using a tokenizer with a fixed vocabulary size. Each word is mapped to a unique integer index based on frequency. Words not present in the vocabulary are replaced with an out-of-vocabulary (OOV) token to handle unseen words.

6. **Sequence Padding:**
   Since movie reviews vary in length, all tokenized sequences are padded or truncated to a fixed maximum sequence length. This ensures uniform input dimensions for batch processing by the neural network models.

7. **Embedding Layer Construction:**
   An embedding layer is added to both the RNN and LSTM models to transform word indices into dense vector representations. These embeddings capture semantic relationships between words and significantly improve model performance compared to sparse representations.

8. **Simple RNN Model Construction:**
   A Simple Recurrent Neural Network (RNN) model is constructed using:
   - An embedding layer.
   - A SimpleRNN layer with dropout and recurrent dropout for regularization.
   - A dense output layer with sigmoid activation for binary sentiment classification.

9. **LSTM Model Construction:**
   An LSTM-based model is constructed using:
   - An embedding layer.
   - An LSTM layer with dropout to reduce overfitting.
   - A sigmoid-activated output layer

   The gating mechanisms in the LSTM enable effective learning of long-term dependencies in textual data.

10. **Model Compilation:**
   Both RNN and LSTM models are compiled using:
    - **Binary Cross-Entropy** as the loss function.
    - **Adam optimizer** with different learning rates for RNN and LSTM training, for efficient       gradient-based optimization.
    - **Accuracy** as the evaluation metric.

11. **Model Training:**
   The RNN and LSTM models are trained for 150 epochs with a batch size of 128. During training:
   - Model checkpointing is used to save the model with the best validation accuracy.
   - Learning rate scheduling is applied using ReduceLROnPlateau to reduce the learning rate when   validation loss stagnates.

12. **Model Selection:**
   After training, the best-performing versions of both models are loaded from the saved checkpoints. This ensures that evaluation is performed using the model state that generalizes best to unseen data.

13. **Model Evaluation:**
   The selected RNN and LSTM models are evaluated on the test dataset. Performance metrics include:
   - Training and validation accuracy trends observed during model training.
   - Test accuracy obtained from the final evaluation.
   - Confusion matrix.
   - Classification report consisting of precision, recall, and F1-score.

14. **Performance Visualization:**
   Learning curves for training and validation loss and accuracy are plotted to analyse model convergence and overfitting behaviour. Additionally, ROC curves and precision–recall curves are generated to assess classification performance more comprehensively.

15. **Performance Comparison:**
   Finally, the performance of the Simple RNN and LSTM models is compared based on accuracy, loss trends, ROC–AUC scores, and classification metrics to highlight the effectiveness of LSTM in handling long-term dependencies in sentiment analysis tasks, by analysing individual curves for both RNN and LSTM, as well as the combined curves.
