Run: 
python run.py

Steps if not doing run.py:
clone repo
run python on bash
import acquire_data
training_data, testing_data = acquire_data.get_data()
import conv_network
net = conv_network.Network(28,5,6,[100,10])
net.model(tr,10,20,0.1,0.5)

Training_data is a list of 10000 arrays of size 28x28

Note: mnist_network.py is simply a fully connected network,
in order to run, update acquire_data to get training and 
testing data to have 784x1 arrays, not 28x28.
