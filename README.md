# 📰 Transformer-Chinese-News-Generator

A lightweight, character-level generative language model based on the **Transformer Decoder** architecture, specifically trained to generate Chinese news text.

## 🔬 Background & Reference

This project is a custom implementation of the architecture introduced in the seminal paper:

> **"Attention Is All You Need"** (Vaswani et al., 2017)

By leveraging **Multi-Head Self-Attention** and **Causal Masking**, this model learns the statistical patterns and linguistic structures of Chinese news reporting to generate coherent, context-aware sentences.

![Transformer architecture diagram，AI 生成](https://encrypted-tbn1.gstatic.com/licensed-image?q=tbn:ANd9GcSvYwO3pBQl7HUDiptIgOyY-yZv8fpyiBn5ohBDGdo-kEZ3t4GzkYNjHjVH617RuQSpWynIJkXZfVnf6RegyOJ8JLh8P7l0_1IMFQNlnYnce1m8Gv8)


---

## 🚀 Key Features

- **Architecture**: A Decoder-only Transformer with $L=4$ layers and $H=4$ attention heads.
    
- **Text Generation Strategy**:
    
    - **Temperature Scaling**: Controls the randomness of the output. Higher values (e.g., 1.0) increase diversity, while lower values (e.g., 0.2) make the model more "confident" and repetitive.
        
    - **Top-K Sampling**: Filters the top $K$ most likely next tokens to prevent the model from picking highly improbable (low-probability) words that cause "gibberish" outputs.
        
- **Efficiency**: Optimized for small-scale datasets with a block size of 64 tokens, making it ideal for rapid prototyping.
    

---

## 💻 Hardware Requirements

This model is highly efficient and can run on most modern consumer hardware.

|**Component**|**Minimum**|**Recommended**|
|---|---|---|
|**GPU**|Optional (Supports CPU)|NVIDIA GPU (GTX 10-series or newer)|
|**VRAM**|N/A|512MB+|
|**RAM**|4GB|8GB|
|**Storage**|< 100MB|100MB+|


## 🛠 Installation & Usage

### 1. Requirements

Ensure you have Python 3.8+ and PyTorch installed:

Bash

```
pip install torch
```

### 2. Files Setup

Keep the following files in your project directory:

- `train.py`: The inference script.
    
- `gpt_ckp_token64.pth`: The pre-trained model weights.
    
- `report.txt`: The source text used to map the vocabulary (chars).
    


---

## 🔍 Technical Implementation Details

The model processes text through the following pipeline:

1. **Tokenization**: Character-level encoding.
    
2. **Embedding**: Combined Token and Positional embeddings.
    
3. **Transformer Blocks**:
    
    - **LayerNorm** (Pre-norm formulation).
        
    - **Multi-Head Attention** with scaled dot-product.
        
    - **Feed-Forward Network** ($4 \times$ expansion).
        
4. **Generation Loop**: Uses a sliding window (context length of 64) to predict the next character iteratively.


# Transformer Architecture & Logic Notes

## 1. Dimensionality and Multi-Head Attention

- **Dimensionality Relationship**: $d_k = d_{query} = d_{key} = d_{value} = d_{embed} / n_{head}$.
    
- **Core Concept**: Queries and Keys act as dimensionality reducers to find abstract semantic relationships. By splitting into **Multi-Head Attention**, each head can focus on different types of feature relationships (e.g., one head for syntax, another for entity relationships).
    
- **Attention Score**: Calculated as $\frac{QK^T}{\sqrt{d_k}}$.
    
- **Weight Tying (Advanced Trick)**: In models like GPT and BERT, the **Embedding Matrix** $[V, d_{embed}]$ (responsible for initial "translation") and the **Unembedding Matrix** at the output layer (responsible for final "token selection") often share the same weights. The output matrix is essentially the transpose of the input matrix.
    

---

## 2. Conceptual Understanding: The Library Analogy

To intuitively grasp the three vectors, imagine searching for a book in a **Library**:

- **Query (Q)**: Your **requirement** or search intent. (e.g., "I need a book on Deep Learning math.")
    
- **Key (K)**: The **labels/titles** on the book spines. The system matches your Query (Q) against all Keys (K).
    
- **Value (V)**: The **actual content** inside the books. Once a match is found between Q and K, you focus on the content (V) of that specific book.
    

---

## 3. Mathematical Definitions & Logic

Every input token (vector $x$) is transformed by three weight matrices ($W^Q, W^K, W^V$):

1. **$Q = X \cdot W^Q$ (Query)**: Represents what the current token is "asking" other tokens.
    
2. **$K = X \cdot W^K$ (Key)**: Represents how this token is "indexed" or categorized.
    
3. **$V = X \cdot W^V$ (Value)**: Represents the actual information contained in this token.
    

### The Scaled Dot-Product Attention Formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Execution Steps**:

1. **Similarity ($QK^T$)**: Dot product of Q and K determines how relevant every token is to the current one. $\sqrt{d_k}$ is the scaling factor to prevent gradient vanishing.
    
2. **Normalization (Softmax)**: Converts scores into **probability weights** (summing to 1). This determines "how much attention" to pay to each token.
    
3. **Weighted Sum (Multiply by $V$)**: Multiplies weights by $V$ to get an output matrix $\Delta X$. n_head ✖️ \[token_length, d_k\] This represents a **residual update** or "refinement" of the token's meaning based on context in a lower-dimensional space ($d_k$).
    
4. **Output Projection ($W^{output}$)**: $\Delta X$ is projected back to $d_{embed}$ and summed across all heads:$\Delta X*W^{output}$  ->\[token_length, d_embed\]
    
    $$X_{out} = X + \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
    

---

## 4. The Processing Workshop: FFN (Feed-Forward Network)

While Attention handles "communication," the **FFN** handles "knowledge processing."

- **Up-Projection ($W_{up}$)**: Expands dimension from $d_{model}$ (e.g., 512) to $4 \times d_{model}$ (e.g., 2048).
    
    - _Why?_ To find a **wider feature space**. Complex logic that "collides" in 512D can be unfolded and processed clearly in 2048D.
        
- **Down-Projection ($W_{down}$)**: Compresses 2048 back to 512, consolidating learned knowledge for the next layer.
    

**Where do the parameters live?**

- **FFN Layers**: Occupy ~**2/3** of parameters. This is where the model stores **factual knowledge**.
    
- **Attention Layers**: Occupy ~**1/3** of parameters. This is where the model stores **reasoning logic**.
    

---

## 5. Architectural Variations: Pre-LN vs. Post-LN

Modern models (GPT-3, Llama) use **Pre-LN**, which differs from the original "Attention Is All You Need" paper (**Post-LN**).

|**Feature**|**Post-LN (Original Paper)**|**Pre-LN (Modern Standard)**|
|---|---|---|
|**Logic**|`Add -> Norm`|`Norm -> Add`|
|**Formula**|$x = \text{LayerNorm}(x + \text{Sublayer}(x))$|$x = x + \text{Sublayer}(\text{LayerNorm}(x))$|
|**Stability**|Harder to train (sensitive LR)|**Highly stable**|
|**Depth**|Limited layers|**Supports 100+ layers**|

---

## 6. Encoder vs. Decoder: The Philosophical Difference

- **Encoder (e.g., BERT)**: Goal is **Understanding**. It is **Bi-directional** (no Mask) because to understand a word like "Apple," you need to see if the next word is "Store" or "Juice."
    
- **Decoder (e.g., GPT)**: Goal is **Generation**. It is **Causal** (uses Mask) because when writing word #5, word #6 hasn't been "born" yet. Looking ahead is "cheating."
    
| **Architecture Type** | **Representative Models** | **Structural Characteristics**                    | **Best Suited For...**                                                  |
| --------------------- | ------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| **Encoder-Decoder**   | T5, BART                  | Bi-directional Encoding + Unidirectional Decoding | Machine Translation, Text Summarization, Error Correction               |
| **Encoder-Only**      | BERT                      | Full Bi-directional Attention (No Masking)        | Text Classification, Named Entity Recognition (NER), Sentiment Analysis |
| **Decoder-Only**      | **GPT Series, Llama**     | **Full Unidirectional Causal Attention**          | **Conversational AI, Logical Reasoning, Zero-shot Learning**            |
**Why Decoder-Only is King?**

1. **Simplicity**: Treats every task (translation, coding, math) as a "Next Token Prediction" game.
    
2. **Efficiency**: Causal masking allows the model to calculate the loss for the entire sequence in one forward pass during training.
    
3. **Inference (KV Cache)**: We store previous $K$ and $V$ values so we only need to compute $Q$ for the single newest token, drastically speeding up generation.
    

---

## 7. Strategy: Temperature & Top-K & Top-P & EOS

During inference, we use hyperparameters to control the "creativity" of the output:

- **Temperature ($T$)**: $\text{probs} = \text{softmax}(\frac{\text{logits}}{T})$
    
    - **High $T$**: Flattens distribution. Result: **Novel but nonsense**.
        
    - **Low $T$**: Sharpens distribution. Result: **Repetitive but reliable**.
        
- **Top-K Sampling**: Only samples from the top $K$ most likely words. This prevents the model from picking "gibberish" words with tiny probabilities.
    
- **EOS (End of Sentence)**: The model stops when it predicts a special `<|endoftext|>` token, which it learned during training to signify the completion of a thought.
    

---

### Training vs. Generation Comparison

|**Dimension**|**Training**|**Generation (Inference)**|
|---|---|---|
|**Input**|Full known sentence|Prompt only|
|**Output**|Softmax + CrossEntropy|Softmax + Sampling (`multinomial`)|
|**Logic**|Parallel computation|Sequential (one-by-one)|
|**Randomness**|Deterministic|Stochastic (Randomized)|
