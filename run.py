import acquire_data
import conv_network
tr, te = acquire_data.get_data()
net = conv_network.Network(28,5,6,[10])
net.model(tr,10,20,0.001,0.05)
