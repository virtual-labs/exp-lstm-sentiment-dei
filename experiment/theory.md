### Theory

#### Introduction to Sentiment Analysis
Sentiment Analysis is a supervised Natural Language Processing (NLP) task that aims to automatically identify and classify the emotional polarity expressed in textual data. The most common form of sentiment analysis focuses on determining whether a given text conveys a positive or negative sentiment. This task plays a crucial role in applications such as opinion mining, customer feedback analysis, recommendation systems, market analysis, and social media monitoring.

In this experiment, sentiment analysis is formulated as a binary classification problem, where the objective is to predict the sentiment label of a movie review based on its textual content:

**y ∈ {0, 1}**

where,
- **0** represents a Negative review,
- **1** represents a Positive review.

Since textual data is inherently sequential, capturing word order and contextual dependencies is
essential for accurate sentiment prediction. Recurrent Neural Networks (RNNs) and their advanced
variants, such as Long Short-Term Memory (LSTM) networks, are particularly well-suited for this task.

---

#### Long Short-Term Memory (LSTM)
Long Short-Term Memory (LSTM) is a specialized type of Recurrent Neural Network (RNN) designed to
overcome the key limitations of traditional RNNs, particularly the vanishing and exploding gradient problems. LSTM networks were introduced by Hochreiter and Schmidhuber in 1997 with the explicit goal of enabling neural networks to learn and retain long-term dependencies in sequential data.

Unlike standard RNNs, which rely solely on a single hidden state, LSTMs introduce an internal cell state that acts as a memory pipeline. This cell state allows information to flow across time steps with minimal modification, making it easier for gradients to propagate during backpropagation through time (BPTT). As a result, LSTMs are capable of remembering important contextual information over long sequences while selectively forgetting irrelevant details.

**LSTM Cell Components:**

An LSTM cell consists of several interacting components that regulate the flow of information using gating mechanisms. These gates are implemented using sigmoid and tanh activation functions, which enable fine-grained control over memory updates.

1.  **Forget Gate (f<sub>t</sub>):** The forget gate determines which information from the previous cell state should be retained or discarded. This process is implemented by a sigmoid activation function. The decision is based on the previous hidden state and the current input. The output of the forget gate is a vector of values between 0 and 1, where values close to 0 indicate information to be forgotten, and values close to 1 indicate information to be retained.
    
    **f<sub>t</sub> = σ(W<sub>f</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>f</sub>)**

2.  **Input Gate (i<sub>t</sub>):** The input gate controls the extent to which new information from the current input should be written to the cell state. This gate protects the memory from being corrupted by irrelevant or noisy inputs and ensures that only meaningful information is incorporated.
    
    **i<sub>t</sub> = σ(W<sub>i</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>i</sub>)**

3.  **Candidate Cell State (C̃<sub>t</sub>):** The candidate cell state is the key to LSTMs and represents the memory of LSTM networks. It represents new information that could potentially be added to the memory. The LSTM block removes or adds information to the cell state through the gates, which allow optional information to cross.
    
    **C̃<sub>t</sub> = tanh(W<sub>c</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>c</sub>)**

4.  **Cell State Update (C<sub>t</sub>):** The cell state update combines the retained information from the previous cell state with the newly generated candidate information. This additive update mechanism is a key innovation of LSTM networks, as it allows gradients to flow across long sequences without rapid decay, enabling effective learning of long-term dependencies.
    
    **C<sub>t</sub> = f<sub>t</sub> · C<sub>t-1</sub> + i<sub>t</sub> · C̃<sub>t</sub>**

5.  **Output Gate (o<sub>t</sub>):** The output gate determines which part of the updated cell state should be exposed as the hidden state for the current time step. This allows the LSTM to control how much of its internal memory influences the output.
    
    **o<sub>t</sub> = σ(W<sub>o</sub>[h<sub>t-1</sub>, p<sub>t</sub>] + b<sub>o</sub>)**

6.  **Hidden State (h<sub>t</sub>):** The hidden state represents the final output of the LSTM cell at time step t. It is obtained by applying a tanh activation to the updated cell state and modulating it with the output gate.
    
    **h<sub>t</sub> = o<sub>t</sub> · tanh(C<sub>t</sub>)**

---

#### LSTM Architecture & Information Flow-

The neural network architecture of an LSTM block, illustrated in Figure 1, demonstrates how LSTM
networks extend the memory capabilities of traditional RNNs. In addition to the hidden state used in RNNs, an LSTM block introduces four additional components: the cell state (C<sub>t</sub>), forget gate (f<sub>t</sub>),input gate (i<sub>t</sub>), and output gate (o<sub>t</sub>), These components interact in a carefully designed manner to regulate the flow of information across time steps. 

The *p<sub>t</sub>*, *h<sub>t-1</sub>*, and *C<sub>t-1</sub>* correspond to the input of the current time step, the hidden output from the previous time step, and the cell state (memory) of the previous unit, respectively. The information from the previous LSTM unit is combined with current input to effectively model sequential patterns and contextual relationships in text data. The LSTM blocks are mainly divided into three gates: forget, input-update, and output. Each of these gates is connected to the cell state to provide the necessary information that flows from the current time step to the next.

![Figure 1- Architecture of LSTM](images/lstm_architecture.png)

**Fig. 1.** Architecture of LSTM.

(Source: H. Okut, “Deep Learning for Subtyping and Prediction of Diseases: Long Short-Term Memory,” IntechOpen).

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