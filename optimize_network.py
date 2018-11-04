import numpy as np
import hyperopt
from functools import partial


# cost function: testing cost
# only testing for 3 layer NN, but could test for 4 and 5 layers

# custom distribution
def log_uniform_int(name, lower, upper):
    return hyperopt.hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

# round to nearest base
def round_optimized(x, base=5):
    return int(base * round(float(x)/base))

# parameter space
parameter_space = {
        'epochs': hyperopt.hp.uniform('epochs', 10, 100),
        'hidden_layers': hyperopt.hp.uniform('hidden_layers', 10, 200),
        'mini_batch_size': log_uniform_int('mini_batch_size', 4, 512),
        'learning_rate': hyperopt.hp.lognormal('learning_rate', 0, 1.5),
        'regularization_parameter': hyperopt.hp.lognormal('regularization_parameter', 0, 0.4),
        'early_stopping_threshold': hyperopt.hp.uniform('early_stopping_threshold', 0.005, 0.25)
}

import acquire_data
training_data, testing_data = acquire_data.get_data()

import mnist_network

# run network and get cost, the objective function to minimize
def create_network(parameters):

    hidden_layers = int(parameters['hidden_layers'])
    layer_sizes = [784, hidden_layers, 10]

    network = mnist_network.Network(layer_sizes)

    epochs = int(parameters['epochs'])
    mini_batch_size = round_optimized(parameters['mini_batch_size'])
    learning_rate = parameters['learning_rate']
    regularization_parameter = parameters['regularization_parameter']
    early_stopping_threshold = parameters['early_stopping_threshold']

    network.model(training_data, epochs, mini_batch_size, learning_rate, regularization_parameter, early_stopping_threshold)
    cost, accuracy = network.predict(testing_data, regularization_parameter)

    return cost

trials = hyperopt.Trials()

# create tree structured parzen estimator object
tpe = partial(hyperopt.tpe.suggest, n_EI_candidates=1000, gamma=0.25, n_startup_jobs=15)

hyperopt.fmin(create_network, trials=trials, space=parameter_space, algo=tpe, max_evals=100)

print trials.best_trial
