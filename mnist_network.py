import sys
import numpy as np
import math

class Network:
    
    def __init__(self, sizes):
        """
        Initializes network. Sizes is just a layer by layer definition of the
        network. Biases are initialized from neurons after input layer using
        standard normal Gaussian distribution. Weights is a list of weight
        matrices from one layer to the next, and are initialized using the
        normal distribution but with standard deviation of 1/sqrt(n) where
        n is the amount of neurons in the input layer. Used to prevent neuron
        saturation in hidden layers to increase learning speed as extreme
        input (z) values to the neurons result in output activations closer 
        to 0 or 1, resulting in smaller gradients and slower learning in the 
        hidden layers.
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.rand(b, 1) for b in self.sizes[1:]]  
        self.weights = [np.random.normal(scale=float(1/np.sqrt(self.sizes[0])), size=(wj,wk)) for wk, wj in zip(self.sizes[0:self.num_layers-1], self.sizes[1:])]

    def model(self, training_data, epochs, mini_batch_size, learning_rate, regularization_parameter):
        
        print "Hyperparameters used - Hidden layers: {0} Epochs: {1} Mini_Batch_Size: {2} Learning Rate: {3} Regularization Parameter: {4}".format(self.sizes[1], 
                epochs, mini_batch_size, learning_rate, regularization_parameter)
       	
        accuracies = [] 
        for epoch in xrange(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k:k+mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size)]
            count = 0
            for batch in batches:
                count += self.update_network(batch, learning_rate, regularization_parameter)
                print count
            
            accuracy = float(count)/len(training_data)
            accuracies.append(accuracy)
            if accuracy < 0.85:
                return 0
            
            epoch_comparison = accuracies[-5:]
            epsilon = 0.001
            if epoch > 10 and max(epoch_comparison)-min(epoch_comparison) < epsilon:
                return 0
            print "Epoch {0}: {1}/{2} correct. {3}% accuracy!".format(epoch+1, count, len(training_data), round(accuracy, 5) * 100)
    
    def update_network(self, mini_batch, learning_rate, regularization_parameter):
        
        b_grad_sum = [np.zeros(b.shape) for b in self.biases]
        w_grad_sum = [np.zeros(w.shape) for w in self.weights]

        count = 0
        for train_data in mini_batch:
            t_input, t_output = train_data
            b_grad, w_grad, prediction = self.backpropagate(t_input, t_output)
            b_grad_sum = [cv+bg for cv,bg in zip(b_grad_sum, b_grad)]
            w_grad_sum = [cv+wg+(regularization_parameter*wg) for cv,wg in zip(w_grad_sum, w_grad)] 
            count+=prediction

        self.biases = [b-((learning_rate*bgs)/len(mini_batch)) for b, bgs in zip(self.biases, b_grad_sum)]
        self.weights = [w-((learning_rate*wgs)/len(mini_batch)) for w, wgs in zip(self.weights, w_grad_sum)]    
        return count

    def backpropagate(self, x, y):
        
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]
 
        z_vectors = []
        activations = [x]
        for layer in xrange(self.num_layers-1):
            a = activations[layer]
            z = np.dot(self.weights[layer], a) + self.biases[layer]
            z_vectors.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        accurate = self.evaluate(activations[-1], y)
        
        # outer layer error initialized to partial derivative of input vector
        # z with respect to cross entropy cost where Cx = -[y ln
        # a + (1-y)ln(1-a)] 
        error_outer_layer = activations[-1] - y
        errors = [error_outer_layer]
        w_grad[-1] = np.dot(error_outer_layer, activations[-2].transpose())
        b_grad[-1] = error_outer_layer

        for layer in xrange(self.num_layers-2,0,-1):
            curr_layer = layer-1
            error = np.dot(self.weights[curr_layer+1].transpose(), errors[-1])*self.sigmoid_derivative(z_vectors[curr_layer])        
            errors.append(error) 
            w_grad[curr_layer] = np.dot(error, activations[curr_layer].transpose())
            b_grad[curr_layer] = error
            
        return (b_grad, w_grad, accurate)

    def evaluate(self, actual, expected):
        return np.argmax(actual) == np.argmax(expected)

    def get_cost(self, actual, expected):
        return np.sum(np.nan_to_num(-1*expected*np.log(actual)-(1-expected)*np.log(1-actual)))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def predict(self, testing_data, regularization_parameter):

        count = 0
        cost = 0.0
        for test in testing_data:
            test_input, test_output = test
            activations = [test_input]
            for layer in xrange(self.num_layers-1):
                a = activations[layer]
                z = np.dot(self.weights[layer], a) + self.biases[layer]
                a = self.sigmoid(z)
                activations.append(a)
            if self.evaluate(activations[-1], test_output):
                count+=1
            c_zero = self.get_cost(activations[-1], test_output)
            cost += c_zero
        c_reg = (regularization_parameter/2) * sum(map(lambda w:np.sum(w**2), self.weights))
        cost += c_reg

        print "Testing Accuracy: {0}/{1} correct. {2}% accuracy!".format(count, len(testing_data), round(float(count)/len(testing_data), 5) * 100) 
	print "Cost: {0}\n".format(cost)
        return (cost, float(count)/len(testing_data))
