# MNIST digits classification using a FFNN from scratch

Using FFNN with: 1 input layer, 1 hidden layer of 20 neurons, 1 output layer of 10 neurons. The data is the MNIST dataset in txt files.

Input layer: (60 000 x 784) with float values between 0 and 1.

Hidden layer: ((784 + 1) x 20), +1 is for the bias, with Sigmoid activation.

Output layer: (60 000 x 10), with Softmax activation.

For now the best model gave me 92,55% on the test, limited to 10000 epochs, in 36min47 (slow Intel i5).

# Mathematics and backpropagation

Here the bias (a column of ones) is added in matrices before being dotted by weights matrices.


For $$k = 1, 2, ..., K$$  and $$j = 1, 2, ..., J$$

$$
w_{kj}^{(t+1)} = w_{kj}^{(t)} - \alpha1 \frac{\partial E}{\partial w_{kj}^{(t)}} 
$$

For $$n = 0, 1, ..., N$$ and $$k = 1, 2, ..., K$$

$$
v_{nk}^{(t+1)} = v_{nk}^{(t)} - \alpha2 \frac{\partial E}{\partial v_{nk}^{(t)}}
$$

where  $$\alpha$$ are learning rates.

<h3>Derivation</h3> 
The Sum of Squared Errors (SSE) is defined as:

$$
E = \frac{1}{2} \sum_{i=1}^{I} \sum_{j=1}^{J} (g_j^{(i)} - y_j^{(i)})^2
$$

Let's derive by $$w_{kj}$$ :

$$
\frac{\partial E}{\partial w_{kj}} = \sum_{i=1}^{I} (g_j^{(i)} - y_j^{(i)}) g_j^{(i)} (1 - g_j^{(i)}) \tilde{ f_j^{(i)}}
$$

Then, let's derive by $$v_{nk}$$ :

$$
\frac{\partial E}{\partial v_{nk}} = \sum_{i=1}^{I} \sum_{j=1}^{J} (g_j^{(i)} - y_j^{(i)}) g_j^{(i)} (1 - g_j^{(i)}) w_{kj} f_k^{(i)} (1 - f_k^{(i)}) \bar{X_n^{(i)}}
$$

Please unzip both .zip to get the corresponding folders.

S/O to my professor Jae Yun JUN KIM.

PS: I changed from SSE to Cross Entropy Loss, I didn't changed the backpropagation yet.
