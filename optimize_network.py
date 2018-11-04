import numpy as np
import hyperopt
from functools import partial


# cost function: testing cost
# epochs: 5,10,20,30,40,50,75,100,200
# hidden layers 2-3 NN: 0, 10, 20, 30, 40, 50, 75, 100, 200, 300, 500  
# mini_batch_size: 1, 5, 10, 25, 50, 75, 100, 500, 1000, 5000, 10000
# learning_rate: 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
#               10.0, 25.0, 50.0, 100.0
# regularization_parameter: 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25. 0.5, 1.0,
#               2.5, 5.0, 10.0, 25.0, 50.0, 100.0

def log_uniform_int(name, lower, upper):
    return hyperopt.hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

def round_optimized(x, base=5):
    return int(base * round(float(x)/base))

parameter_space = {
        'epochs': hyperopt.hp.uniform('epochs', 10, 100),
        'hidden_layers': hyperopt.hp.uniform('hidden_layers', 10, 200),
        'mini_batch_size': log_uniform_int('mini_batch_size', 4, 512),
        'learning_rate': hyperopt.hp.lognormal('learning_rate', 0, 1.0),
        'regularization_parameter': hyperopt.hp.lognormal('regularization_parameter', 0, 0.5)
}

import acquire_data
training_data, testing_data = acquire_data.get_data()

import mnist_network

def create_network(parameters):

    hidden_layers = int(parameters['hidden_layers'])
    layer_sizes = [784, hidden_layers, 10]
    
    network = mnist_network.Network(layer_sizes)

    epochs = int(parameters['epochs'])
    mini_batch_size = round_optimized(parameters['mini_batch_size'])
    learning_rate = parameters['learning_rate']
    regularization_parameter = parameters['regularization_parameter']

    network.model(training_data, epochs, mini_batch_size, learning_rate, regularization_parameter) 
    cost, accuracy = network.predict(testing_data, regularization_parameter)

    return cost

trials = hyperopt.Trials()
tpe = partial(hyperopt.tpe.suggest, n_EI_candidates=1000, gamma=0.25, n_startup_jobs=15)

hyperopt.fmin(create_network, trials=trials, space=parameter_space, algo=tpe, max_evals=100)

print trials.best_trial
