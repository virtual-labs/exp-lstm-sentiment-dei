### Theory

#### Introduction to Sentiment Analysis
Sentiment Analysis is a supervised natural language processing (NLP) task that determines the emotional polarity (positive or negative) expressed in text. It is widely used in applications such as opinion mining, recommendation systems, and social media analysis.

In this experiment, sentiment analysis is formulated as a binary classification problem:

**y ∈ {0, 1}**

where,
- **0** = Negative review
- **1** = Positive review

---

#### Long Short-Term Memory (LSTM)
Long Short-Term Memory (LSTM) is a specialized type of Recurrent Neural Network (RNN) designed to
overcome the key limitations of traditional RNNs, particularly the vanishing and exploding gradient problems. LSTM networks were introduced by Hochreiter and Schmidhuber in 1997 with the explicit goal of enabling neural networks to learn and retain long-term dependencies in sequential data.

Unlike standard RNNs, which rely solely on a single hidden state, LSTMs introduce an internal cell state that acts as a memory pipeline. This cell state allows information to flow across time steps with minimal modification, making it easier for gradients to propagate during backpropagation through time (BPTT). As a result, LSTMs are capable of remembering important contextual information over long sequences while selectively forgetting irrelevant details.

**LSTM Cell Components:**

An LSTM cell consists of several interacting components that regulate the flow of information using gating mechanisms. These gates are implemented using sigmoid and tanh activation functions, which enable fine-grained control over memory updates.

1.  **Forget Gate (f<sub>t</sub>):** It decides the type of information that should be thrown away or kept from the cell state. This process is implemented by a sigmoid activation function.
    
    **f<sub>t</sub> = σ(W<sub>f</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>f</sub>)**

2.  **Input Gate (i<sub>t</sub>):** It controls what new information will be added to the cell state from the current input. This gate also plays the role to protect the memory contents from perturbation by irrelevant input.
    
    **i<sub>t</sub> = σ(W<sub>i</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>i</sub>)**

3.  **Candidate Cell State (C̃<sub>t</sub>):** This is the key to LSTMs and represents the memory of LSTM networks. The LSTM block removes or adds information to the cell state through the gates, which allow optional information to cross.
    
    **C̃<sub>t</sub> = tanh(W<sub>c</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>c</sub>)**

4.  **Cell State Update (C<sub>t</sub>):** This additive update mechanism allows gradients to flow across long sequences, enabling long-term dependency learning.
    
    **C<sub>t</sub> = f<sub>t</sub> · C<sub>t-1</sub> + i<sub>t</sub> · C̃<sub>t</sub>**

5.  **Output Gate (o<sub>t</sub>):** It controls which information to reveal from the updated cell state (C<sub>t</sub>) to the output in a single time step. In other words, the output gate determines what the value of the next hidden state should be in each time step.
    
    **o<sub>t</sub> = σ(W<sub>o</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>o</sub>)**

6.  **Hidden State (h<sub>t</sub>):** The hidden state represents the output of the LSTM cell at time step *t*. It is computed by applying a non-linear transformation to the updated cell state and modulating it with the output gate.
    
    **h<sub>t</sub> = o<sub>t</sub> · tanh(C<sub>t</sub>)**

---

#### LSTM Architecture & Information Flow-

The neural network architecture for an LSTM block given in Figure 1 demonstrates that the LSTM network extends RNN's memory and can selectively remember or forget information by structures called cell states and three gates. Thus, in addition to a hidden state in RNN, an LSTM block typically has four more layers. These layers are called the cell state (C<sub>t</sub>), an input gate (i<sub>t</sub>), an output gate (o<sub>t</sub>), and a forget gate (f<sub>t</sub>). Each layer interacts with each other in a very special way to generate information from the training data.

The *p<sub>t</sub>*, *h<sub>t-1</sub>*, and *C<sub>t-1</sub>* correspond to the input of the current time step, the hidden output from the previous LSTM unit, and the cell state (memory) of the previous unit, respectively. The information from the previous LSTM unit is combined with current input to generate a newly predicted value. The LSTM blocks are mainly divided into three gates: forget, input-update, and output. Each of these gates is connected to the cell state to provide the necessary information that flows from the current time step to the next.

![Figure 1- Architecture of LSTM](images/lstm_architecture.png)

**Fig. 1.** Architecture of LSTM.

Source: H. Okut, “Deep Learning for Subtyping and Prediction of Diseases: Long Short-Term Memory,” IntechOpen).

Due to these properties, LSTMs are widely used in applications such as sentiment analysis, machine translation, speech recognition, and time-series forecasting, where understanding long-range dependencies is critical for accurate predictions.

---

#### Merits of Long Short-Term Memory

-   **Handles Long-Term Dependencies:** LSTM networks are capable of learning long-term dependencies in sequential data, overcoming the vanishing gradient problem present in traditional RNNs.
-   **Effective Memory Management:** The use of forget, input, and output gates allows LSTM to selectively store, update, or discard information, leading to better sequence modelling.
-   **Suitable for Sequential Data:** LSTMs perform well on time-series, text, speech, and sentiment analysis tasks where data has temporal dependencies.
-   **Stable Training:** Due to controlled gradient flow through the cell state, LSTMs provide more stable training compared to simple RNNs.
-   **Better Performance on Contextual Tasks:** LSTMs capture contextual information over longer sequences, improving performance in tasks such as language modelling and text classification.

#### Demerits of Long Short-Term Memory

-   **High Computational Cost:** LSTM networks involve multiple gate computations, making them computationally expensive compared to standard RNNs.
-   **Longer Training Time:** Due to their complex structure, LSTMs require more time to train, especially on large datasets.
-   **Large Memory Requirement:** The presence of multiple weight matrices and gates increases memory consumption.
-   **Risk of Overfitting on Small Datasets:** When trained on small datasets, LSTMs may overfit if proper regularization techniques are not applied.
-   **Complex Architecture:** The internal structure of LSTM cells makes them harder to understand, tune, and debug compared to simpler models.