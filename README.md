# Neural Network Multi-Classifier (NumPy Only)
--
## Intro
This project implements a **multi-layer neural network classifier** from
scratch using only **NumPy** (without deep learning frameworks).\
It demonstrates the fundamentals of **feed-forward neural networks**,
**backpropagation**, and **gradient descent optimization** on sample
image data.

------------------------------------------------------------------------

## Features

-   Pure NumPy implementation (no TensorFlow/PyTorch).
-   3-layer feedforward neural network:
    -   Input layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer.
-   **Tanh** activation for hidden layers.
-   **Softmax** activation for the output layer.
-   **Cross-Entropy Loss** for training.
-   Mini-batch **Stochastic Gradient Descent (SGD)**.
-   Accuracy evaluation on test data.

------------------------------------------------------------------------

## Project Structure

    .
    ├── input_output/
    ├── sample/
    │   ├── X.npy   # Input features (normalized images)
    │   ├── y.npy   # Labels
    ├── 1_hw_softmax_batch_template.py    
    ├── 2_hw_crossentropy_batch_template.py     
    ├── 3_hw_feedforward_template.py     
    ├── 4_hw_backwardforward_template.py  
    ├── 5_hw_multiclassifier_template.py    Full Network
    

------------------------------------------------------------------------

## Requirements

Install dependencies before running:

``` bash
pip install numpy scikit-learn
```

------------------------------------------------------------------------

## Usage

1.  Place your dataset inside the `./sample/` folder:
    -   `X.npy`: Feature matrix (images flattened).
    -   `y.npy`: Labels (digits, categories, etc.).
2.  Run the training script:

``` bash
python 5_hw_multiclassifier_template.py
```

------------------------------------------------------------------------

## Example Output

During training, the script prints the **loss** and **accuracy** after
each epoch:

    Epoch 1, Last Loss: 2.301, Acc: 0.112
    Epoch 2, Last Loss: 2.145, Acc: 0.217
    ...
    Epoch 20, Last Loss: 0.452, Acc: 0.876

------------------------------------------------------------------------

## How It Works

1.  **Feedforward Pass**
    -   Computes activations through hidden layers using `tanh`.
    -   Applies `softmax` for multi-class probabilities at the output
        layer.
2.  **Backpropagation**
    -   Computes gradients of weights and biases using the chain rule.
    -   Updates parameters with gradient descent.
3.  **Evaluation**
    -   Uses `accuracy_score` from `sklearn` to measure test
        performance.

------------------------------------------------------------------------

##  Author

Developed by [**Mohamed Saad**](https://www.linkedin.com/in/ibnsa3d/) 
