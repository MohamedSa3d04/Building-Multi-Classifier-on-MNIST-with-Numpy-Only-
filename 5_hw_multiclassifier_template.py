import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from numpy.linalg import norm


np.random.seed(0)


def load_data():
    X = np.load('./sample/X.npy')
    y = np.load('./sample/y.npy')

    # Normalize data
    X = X / 255.0

    return X, y

def tanh(x):
    ex = np.exp(x)
    e_negative_x = np.exp(-x)
    return (ex - e_negative_x) / (ex + e_negative_x)

def dtanh(y):   # tanh derivative
    return 1 - y ** 2


def softmax_batch(x):
    optimal_x = x - np.max(x, axis = 1, keepdims=True)
    return np.exp(optimal_x) / np.sum(np.exp(optimal_x), axis=1, keepdims=True)


def cross_entropy_batch(y_true, y_pred):
    return np.mean(-np.sum(y_true * np.log(y_pred + 1e-15), axis=1))


class NeuralNetworkMultiClassifier:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))

        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((1, hidden_dim2))

        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    def train(self, X_train, y_train, X_test, y_test, learning_rate = 1e-2, n_epochs = 20, batch_size = 32):
        # Feed Forword
        def feed_forward(X_batch):
            net1 = np.dot(X_batch, self.W1) + self.b1
            out1 = tanh(net1)      # out: 32X20
            net2 = np.dot(out1, self.W2) + self.b2       
            out2 = tanh(net2)      # out: 32X15
            net3 = np.dot(out2, self.W3) + self.b3    
            out3 = softmax_batch(net3) # out: 32X10
            return out1, out2, out3


        # BackProbagation
        def backward(X_batch, y_batch, W2, W3, out1, out2, out3):
            dE_dnet3 = out3 - y_batch   
            dE_dout2 = np.dot(dE_dnet3, W3.T)      
            dE_dnet2 = dE_dout2 * dtanh(out2)  
            dE_dout1 = np.dot(dE_dnet2, W2.T)      
            dE_dnet1 = dE_dout1 * dtanh(out1)  

            dW3 = np.dot(out2.T, dE_dnet3)  
            db3 = np.sum(dE_dnet3, axis=0, keepdims=True)   # b3: 1X15
            dW2 = np.dot(out1.T, dE_dnet2)     
            db2 = np.sum(dE_dnet2, axis=0, keepdims=True)        
            dW1 = np.dot(X_batch.T, dE_dnet1)       
            db1 = np.sum(dE_dnet1, axis=0, keepdims=True)

            return dW1, db1, dW2, db2, dW3, db3

        #Update Weights
        def update_weights(dw1, db1, dw2, db2, dw3, db3):
            self.W1 -= dw1 * learning_rate
            self.W2 -= dw2 * learning_rate
            self.W3 -= dw3 * learning_rate
            self.b1 -= db1 * learning_rate
            self.b2 -= db2 * learning_rate
            self.b3 -= db3 * learning_rate

        # Calling functions
        for i in range(n_epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffeld = X_train[permutation]
            y_shuffeld = y_train[permutation]
            for b in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffeld[b:b+batch_size]
                y_batch = y_shuffeld[b:b+batch_size]
                out1, out2, out3 = feed_forward(X_batch)
                
                dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, y_batch, self.W2, self.W3, out1 \
                                                        ,out2 , out3)
                
                update_weights(dW1, db1, dW2, db2, dW3, db3)
            predicted = feed_forward(X_shuffeld)[-1]      
            print(f'Epoch {i+1}, Last Loss: {cross_entropy_batch(y_shuffeld, predicted)}, Acc: {self.accuracy(feed_forward, X_test, y_test)}')

    
    def accuracy(self, forward, X_test, y_test):
        predicted = forward(X_test)[-1]
        return accuracy_score(np.argmax(y_test, axis=1), np.argmax(predicted, axis=1))

if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    nn = NeuralNetworkMultiClassifier(X_train.shape[1], 20, 15, 10)

    nn.train(X_train, y_train, X_test, y_test)
