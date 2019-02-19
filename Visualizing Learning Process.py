import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Sig moid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#Creating evenly spaced values to be trained on
val = np.linspace(-5.0, 5.0, 1000).reshape(-1, 1)

#tensorflow variable for weights and biases
#seed = 1 for reproducibility
w1 = tf.get_variable("w1", shape=[1, 2], initializer = tf.glorot_uniform_initializer(seed=1))
b1 = tf.get_variable("b1", shape=[2], initializer = tf.glorot_uniform_initializer(seed=1))

w2 = tf.get_variable("w2", shape=[2, 1], initializer = tf.glorot_uniform_initializer(seed=1))
b2 = tf.get_variable("b2", shape=[1], initializer = tf.glorot_uniform_initializer(seed=1))

#Input placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 1])

#Creating single layer network
def network(x):
    h1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    out_value = tf.matmul(h1, w2) + b2
    return out_value

#Tensorflow graph operations
sig = network(x)
loss = tf.losses.mean_squared_error(sigmoid(val), sig)
opt = tf.train.AdamOptimizer(0.05).minimize(loss)

#Creating figure for visualization
fig = plt.gcf()
fig.show()
fig.canvas.draw()

#Creating session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ans = sess.run(sig, feed_dict = {x : val})

#Training loop
for i in range(400):
    ans, loss_value, _ = sess.run([sig, loss, opt], feed_dict = {x : val})
    fig.clear()
    #Simoultaneously updating graph
    plt.title("Approximating Sigmoid \n Iteration : {}, Loss : {:.5f}".format(i+1, loss_value))
    plt.ylabel("Output")
    plt.xlabel("Input Values")
    plt.plot(val, sigmoid(val), label="Original", linewidth=2.0)
    plt.plot(val, ans, c = 'r', label = "Network Output")
    plt.legend(loc = 'upper left')
    fig.canvas.draw()
sess.close()

