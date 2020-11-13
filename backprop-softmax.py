
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_d(x):
    return np.exp(x)/(1+np.exp(x))**2
def relu(x):
    return np.maximum(x,0) 
     
def relu_d(x):
    return (x>0).astype('float64')
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu 
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ 
        Set first activation in input layer equal to the input vector x (a 24x24 picture), 
        feed forward through the layers, then return the activations of the last layer.
        Tested: Applied the forward function to the first array in self.trainX and got a result which summed to 1
        All self.a had contents.
        All self.z vectors had contents
        Could all test on a random vector set on a certain seed?
        
        Issues: When applying this function to the array of the first image the values in the array arent between
        0 and 1, this results in the input not being centerd between [-0.5,0.5]
        """
        self.a[0] = x/255 - 0.5      # Center the input values between [-0.5,0.5]
        for i in range(1,self.L-1): 
            self.z[i] = self.w[i]@self.a[i-1]+self.b[i] #Updates all z vectors other than the first and the last
            self.a[i] = self.phi(self.z[i]) #Updates all of the a vectors other than the first and the last 
        self.z[-1] = self.w[-1]@self.a[-2]+self.b[-1] #Updates the final layer of z 
        self.a[-1] = self.softmax(self.z[-1]) #Updates the final layer of a using the softmax function
        return self.a[self.L-1] #Returns the output layer (Probabilities)

    def softmax(self, z):
        """
        Calculates a probability vector from the activation units in the softmax layer.
        Tested on the same array as in the coursework and gets the same results.
        """
        Q = np.sum(np.exp(z)) # Calculates the normalising factor for the softmax function
        return np.exp(z)/Q # Returns the softmax function                 

    def loss(self, pred, y):
        """
        Calculates the partial loss for a given training value (i.e Calculates C(i) for some i)
        Tested using the erxample in the coursework and I got different results.
        I can see how the result he got works but it would indicate that there is a different self.trainY[0].
        """
        return -np.log(pred[np.argmax(y)])
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
             
        """
        self.delta[-1] = self.a[-1] - y
        # Loop to calculate the local gradients:
        for i in range(2,self.L): #Calculates the gradients for all layers 
                    self.delta[-i] = np.multiply(self.phi_d(self.z[-i]),np.dot(self.w[1-i].T,self.delta[1-i]))
                    #Matrix form of calculating the local gradient for the hidden layers
        
        # Loop to calculate the partial derivatives:
        self.db = self.delta
        for i in range(self.L-1): #Sets all the partial derivatives for the weights using the outer products
            self.dw[i+1] = np.outer(self.delta[i+1],self.a[i])
        
     
        
    
    
    def predict(self, x):
        self.forward(x)
        return np.argmax(self.a[-1])

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[-1][j-1]*100 
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            epochs=5):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                # TODO
                batch_dw = [np.zeros(w.shape) for w in self.w]
                batch_db = [np.zeros(b.shape) for b in self.b]
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    self.forward(x)
                    
                    # Compute gradients
                    self.backward(x,y)
                    

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                        batch_db[l] += self.db[l]
                        batch_dw[l] += self.dw[l]
                        
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = self.w[l] - epsilon*(batch_dw[l]/batch_size)
                    self.b[l] = self.b[l] - epsilon*(batch_db[l]/batch_size)
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)


# Start training with default parameters.

def main():
    start_time = time.time()
    bp = BackPropagation()
    bp.sgd()
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    aa = bp.evaluate(bp.testX, bp.testY, 1000)
    print (aa)

if __name__ == "__main__":
    main()
    
