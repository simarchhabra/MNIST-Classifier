import numpy as np
import copy

'''
CNN Structure: 
    Image: (28,28)
    Kernel (Shared Weights): Num_Mapsx5x5
    Convolution Layer: (Num_Maps, 24, 24)
    Pooling Layer (Max Pool 2x2): (Num_Maps, 12, 12)
    Linearized FC (Num_Maps*144, [Hidden Layer Size,10])
    Example: (6,24,24)->(6,12,12)->(864,1)->(100,1)->(10,1)->Softmax Output(10,1)
    
Activation Function: Relu 
Parameters updated after backpropagation: fc_weights/biases, kernel_weights/biases
Code Structure:
    Create Network - Define input_dim (28), feature map size, num maps, sizes
        of fully_connected_layers (no need to include linearized pool layer)
    Model - Get Training Data, Define number of epochs and mini batch size,
            Set learning rate and regularization param for weight decay
    Update Network - Update params after each mini batch
'''

class Network:
    def __init__(self, input_dim, fm_size, fm_num, fully_connected_sizes):
        '''
        Input Dim = 28, fm_size is size of kernel, fm_num is num_maps,
        fully_connected_sizes is list of fully connected layers not including
        linearized pool layer - i.e. [100,10]
        '''
        self.fm_size = fm_size
        self.num_maps = fm_num
        # init kernel weights and biases
        self.kernel_weights = [np.random.randn(fm_size,fm_size) for fm in xrange(self.num_maps)]
        self.kernel_biases = [np.random.randn(1,1) for fm in xrange(self.num_maps)]
        # store fully connected sizes
        self.fc_sizes = fully_connected_sizes
        # get size of linearized pooling layer to create weights 
        pooling_linearized_size = (((input_dim-fm_size+1)**2)*self.num_maps)/4
        self.fc_sizes.insert(0, pooling_linearized_size)
        fc_layers = len(fully_connected_sizes)
        # init fully connected weights and biases
        # i.e. if num_maps is 6, fm_size is 5, hidden layer of 100, output
        # layer of 10, then weights = [array->shape(100,864),array->(10,100)]
        self.fc_weights = [np.random.normal(scale=float(1/np.sqrt(self.fc_sizes[0])), size=(wj,wk)) for wk, wj in zip(self.fc_sizes[0:fc_layers-1], self.fc_sizes[1:])]
        # biases only done for hidden layer and output layer, not pooled
        # linearized layer
        self.fc_biases = [np.random.rand(b, 1) for b in self.fc_sizes[1:]]

    def model(self, training_data, epochs, mini_batch_size, learning_rate, regularization_parameter):
        
        for epoch in xrange(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k:k+mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size)]
            count = 0
            done = 0
            for batch in batches:
                count += self.update_network(batch, learning_rate, regularization_parameter)
                done += len(batch)
                print count, done # print correct amount, inputs gone through for each epoch

            accuracy = float(count)/len(training_data)
            print "Epoch {0}: {1}/{2} correct. {3}% accuracy!".format(epoch+1, count, len(training_data), round(accuracy, 5) * 100)

    def update_network(self, mini_batch, eta, lmbda):
        '''
        Handles feedforwarding, backpropagation and parameter updates for each
        mini batch
        '''
        correct_count = 0 # counts how many correct in mini batch
        
        # sum of fully connected bias and weight deltas
        fc_db_sum = [np.zeros(b.shape) for b in self.fc_biases]
        fc_dw_sum = [np.zeros(w.shape) for w in self.fc_weights]
        # sum of kernel bias and weight deltas
        kernel_db_sum = [np.zeros(b.shape) for b in self.kernel_biases]
        kernel_dw_sum = [np.zeros(w.shape) for w in self.kernel_weights]

        for train_data in mini_batch:
            image, label = train_data
            
            # feedforward
            convolutional_layers = self.convolution(image)
            pooling_layers = self.pool(convolutional_layers)
            fc_z, fc_a = self.fully_connect(pooling_layers)
            output = self.softmax_output(fc_z[-1])
            correct_count += evaluate(output,label) 
            
            # backpropagate
            fc_db, fc_dw, linearized_fc_error = self.bp_fully_connect(output, label, fc_z, fc_a)
            pooling_rows, pooling_cols = pooling_layers[0].shape
            unlinearized_fc_error = linearized_fc_error.reshape((self.num_maps, pooling_rows, pooling_cols))
            pool_error = self.bp_pool(unlinearized_fc_error, convolutional_layers)
            kernel_db, kernel_dw = self.bp_convolution(pool_error, image)
            
            # add parameters for each input, account for weight decay in fully connected layer
            fc_db_sum = [cv+bg for cv,bg in zip(fc_db_sum, fc_db)]
            fc_dw_sum = [cv+wg+(lmbda*wg) for cv,wg in zip(fc_dw_sum, fc_dw)]
            kernel_db_sum = [cv+bg for cv,bg in zip(kernel_db_sum, kernel_db)]
            kernel_dw_sum = [cv+wg for cv,wg in zip(kernel_dw_sum, kernel_dw)]

        # update parameters after completion of batch
        self.fc_biases = [b-((eta*bgs)/len(mini_batch)) for b,bgs in zip(self.fc_biases, fc_db_sum)]
        self.fc_weights = [w-((eta*wgs)/len(mini_batch)) for w,wgs in zip(self.fc_weights, fc_dw_sum)]
        self.kernel_biases = [b-((eta*bgs)/len(mini_batch)) for b,bgs in zip(self.kernel_biases, kernel_db_sum)]
        self.kernel_weights = [w-((eta*wgs)/len(mini_batch)) for w,wgs in zip(self.kernel_weights, kernel_dw_sum)]
        return correct_count

    def convolution(self, image):
        '''
        Feedforward for convolution, returns convolutional layers
        '''
        convolutional_layers = []
        convolution_dim = len(image)-self.fm_size+1
        for i in xrange(self.num_maps):
            convolutional_layer = np.zeros((convolution_dim, convolution_dim))
            for j in xrange(convolution_dim):
                for k in xrange(convolution_dim):
                    input_row_end = j+self.fm_size
                    input_col_end = k+self.fm_size
                    convolutional_layer[j][k] = np.sum(self.kernel_weights[i]*image[j:input_row_end, k:input_col_end])+self.kernel_biases[i]
            convolutional_layers.append(relu(convolutional_layer))
        return convolutional_layers

    def pool(self, convolutional_layers):
        '''
        Max pool feedforward, return pooling layers
        '''
        pooling_layers = []
        for layer in convolutional_layers:
            pooling_layer = [np.max(layer[i:i+2,j:j+2]) for i in xrange(0,len(layer),2) for j in xrange(0,len(layer),2)]
            pooling_layer = np.asarray(pooling_layer).reshape(len(layer)/2,len(layer)/2)
            pooling_layers.append(pooling_layer)
        return pooling_layers
    
    def fully_connect(self, pooling_layers):
        '''
        Feedforward for fully connected layers, linearizes pool layers, and
        computes z vectors (values in activations before activation function),
        along with activations. Z vectors does not include linearized pool
        layer but activations vectors does. Z vectors include output layer,
        activations does not. Returns tuple of z vectors and activations
        '''
        fc_linearized = np.concatenate((np.concatenate(pooling_layers, axis=0)), axis=0)
        z_vectors = []
        activations = [np.asarray(fc_linearized).reshape((len(fc_linearized),1))]
        for layer in xrange(len(self.fc_weights)):
            a = activations[layer]
            z = np.dot(self.fc_weights[layer], a) + self.fc_biases[layer]
            z_vectors.append(z)
            if layer is not len(self.fc_weights)-1:
                a = relu(z)
                activations.append(a)
        return (z_vectors, activations)

    def softmax_output(self, activation):
        '''
        Softmax Layer
        '''
        return np.exp(activation)/np.sum(np.exp(activation))

    def cost(self, output, expected):
        '''
        Negative log likelihood cost and regularization for fc layer
        '''
        return ((-1*np.log(output[np.argmax(expected)]))+np.sum(self.fc_weights*self.fc_weights)

    def bp_fully_connect(self, output, expected_output, z_vectors, activations):
        '''
        Backpropagation of fully connected layer. Returns deltas for fully
        connected weights and biases, along with error for linearized layer.
        Error is defined as derivative of cost with respect to z vector value.
        i.e. partial C/partial Z
        '''
        db = [np.zeros(b.shape) for b in self.fc_biases]
        dw = [np.zeros(w.shape) for w in self.fc_weights]

        # dc/dz of outer layer
        error_outer_layer = output-expected_output
        errors = [error_outer_layer]
        dw[-1] = np.dot(error_outer_layer, activations[-1].transpose())
        db[-1] = error_outer_layer

        for layer in xrange(len(activations)-1,0,-1):
            error = np.dot(self.fc_weights[layer].transpose(), errors[-1])*relu_prime(z_vectors[layer-1])
            errors.append(error)
            dw[layer-1] = np.dot(error, activations[layer-1].transpose())
            db[layer-1] = error

        # dc/dz of pooled linearized layer
        error_input = np.dot(dw[0].transpose(), errors[-1])
        return (db,dw,error_input)

    def bp_pool(self, unlinearized_fc_error, convolutional_layers):
        '''
        Backpropagation of max pool layer, returns error for pooled layer (so
        shape of convolution layers)
        '''
        conv_row, conv_col = convolutional_layers[0].shape
        pool_error = np.zeros((self.num_maps,conv_row,conv_col))
        for layer in xrange(len(convolutional_layers)):
            for i in xrange(0,len(convolutional_layers[layer]),2):
                for j in xrange(0,len(convolutional_layers[layer]),2):
                    row, col = np.unravel_index(np.argmax(convolutional_layers[layer][i:i+2,j:j+2]),(2,2))
                    pool_row = i+row
                    pool_col = j+col
                    pool_error[layer][pool_row][pool_col] = unlinearized_fc_error[layer][i/2][j/2]
        return pool_error

    def bp_convolution(self, pooling_grad, image):
        '''
        Backpropagation of convolution layer, returns deltas of kernel weights
        and biases
        '''
        convolution_error = copy.deepcopy(pooling_grad)
        convolution_error = relu_prime(convolution_error)
        kernel_dw = np.zeros((self.num_maps, self.fm_size, self.fm_size))
        kernel_db = np.zeros((self.num_maps, 1))
        for layer in xrange(len(kernel_dw)):
            for i in xrange(len(kernel_dw[layer])):
                for j in xrange(len(kernel_dw[layer])):
                    ewm_offset = len(image)-len(kernel_dw[0])+1
                    error_weight_matrix = image[i:i+ewm_offset,j:j+ewm_offset]
                    kernel_dw[layer][i][j] = np.sum(error_weight_matrix*convolution_error[layer])
            kernel_db[layer] = np.sum(convolution_error[layer])
        return kernel_db, kernel_dw

def relu(z):
    return np.maximum(z,0)

def relu_prime(z):
    return np.heaviside(z,0)

def evaluate(actual, expected):
    return np.argmax(actual) == np.argmax(expected)
