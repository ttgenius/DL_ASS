# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)

Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)
Neural networks process inputs using small weight values and inputs with large integer values can disrupt or slow down the learning process. Hence we normalize pixel values to use a common scale without distorting differences in the ranges of values or losing information (image can still be viewed normally with pixel values 0-1). And this reduces the dependence of gradients on the scale of the parameters or of their initial values and minimises the potential issue of slow convergence.

Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)
Bias vector is added to increase the flexibility of the model to fit the data. Specifically, it allows the network to better fit the data when all input features are equal to zero, and very likely decreases the bias of the fitted values elsewhere in the data space.

Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)
Backpropagation finds the derivative of every parameter eg.each weight with respect to the error between expectations and predictions, telling the gradient descent how it should update its weight by adjusting the weight gradient slope in the right direction and hence minimise the total error. The two functions represent distinct stages and can be stacked with different optimization schemes. By separating them, we allow one function to be modified without interrupting the other.

Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)
1.The MNIST dataset contains a training set consisting of thousands of images so a good coverage of samples and it is correctly classified with labels that will help us adjust the prediction model.
2.The dataset also contains a test set taken from a different set of people than the original training data, which gives us confidence that our system can recognize digits from people whose writing it didn’t see during training.
3. The dataset images are all pre-aligned (e.g. each image only contains a hand-drawn digit), that the images all have the same square size of 28×28 pixels, and that the images are grayscale, which reduces the variances that could affect our prediction.

Q6: Suppose you are an administrator of the US Postal Service in the 1990s. What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize handwritten zip codes on mail? (2-4 sentences)
Positive: Automate and speed up the repetitive/manual process of recognizing/registering/sorting large amount of zip codes for distribution. Less labour, faster delivery and happier customers.
Negative: Some zip codes may be poorly written and hence incorrectly classified by the MINIST-trained neural network, causing the mail to be delivered incorrectly.

