/**
 * LSTM Sentiment Analysis Simulation - Data
 * Contains pre-computed experiment results from the Jupyter notebook
 */

const EXPERIMENT_DATA = {
    // Parameters
    VOCAB_SIZE: 10000,
    MAX_LEN: 300,
    EMBED_DIM: 256,
    RNN_UNITS: 256,
    LSTM_UNITS: 64,
    BATCH_SIZE: 128,
    EPOCHS: 150,

    // Dataset info
    dataset: {
        train_val_total: 8500,
        test_total: 1500,
        train_size: 7000,
        val_size: 1500,
        test_size: 1500,
        train_balance: [3500, 3500],
        val_balance: [750, 750],
        test_balance: [750, 750],
        vocab_size: 10000
    },

    // Training output
    training_output: `Starting training...

Training RNN...

Training LSTM...

Training completed.

Best models loaded for both RNN and LSTM.`,

    // Accuracy results
    accuracy: {
        rnn: {
            train: 90.76,
            val: 79.47,
            test: 78.87
        },
        lstm: {
            train: 96.17,
            val: 87.33,
            test: 84.80
        }
    },

    // Classification reports
    classification_report: {
        rnn: `              precision    recall  f1-score   support

           0     0.7954    0.7773    0.7862       750
           1     0.7823    0.8000    0.7910       750

    accuracy                         0.7887      1500
   macro avg     0.7888    0.7887    0.7886      1500
weighted avg     0.7888    0.7887    0.7886      1500`,
        lstm: `              precision    recall  f1-score   support

           0     0.8329    0.8707    0.8514       750
           1     0.8645    0.8253    0.8445       750

    accuracy                         0.8480      1500
   macro avg     0.8487    0.8480    0.8479      1500
weighted avg     0.8487    0.8480    0.8479      1500`
    },

    // Image paths for different visualization options
    images: {
        lstm: {
            learning_curves: './images/learning_curves_lstm.png',
            roc_pr: './images/roc_pr_lstm.png',
            confusion_matrix: './images/confusion_matrix_lstm.png'
        },
        rnn: {
            learning_curves: './images/learning_curves_rnn.png',
            roc_pr: './images/roc_pr_rnn.png',
            confusion_matrix: './images/confusion_matrix_rnn.png'
        },
        combined: {
            training_curves: './images/combined_training_curves.png',
            validation_curves: './images/combined_validation_curves.png',
            roc_pr: './images/combined_roc_pr.png'
        }
    }
};
