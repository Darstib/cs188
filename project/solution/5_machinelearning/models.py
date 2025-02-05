import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # return nn.DotProduct(self.w, x)
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        def train_once(dataset):
            """
            Train the perceptron for one epoch.
            Returns True if the perceptron learned something new in this epoch, False otherwise.
            """
            error = False
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                label = nn.as_scalar(y)
                if prediction != label:
                    self.w.update(x, label)
                    error = True
            return error

        MAX_EPOCHS = 10001
        for _ in range(MAX_EPOCHS):
            if _ == MAX_EPOCHS - 1:
                raise Exception("Maximum number of epochs reached.")
            if not train_once(dataset):
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    Supports constructing a network with an arbitrary number of layers.
    """
    def __init__(self):
        # Initialize model parameters for an arbitrary-layer neural network
        self.parameter_num = 1 # in this case is len([x])=1
        self.lr = 0.01
        self.accuracy = 0.02
        self.batch_size = 200 # the smaller the batch size, the greater the difference in accuracy between the training set and the test set.
        self.layer_num = 2 # the bigger the layer number, the faster each epoch but more epoches.
        self.layer_size = 256 # the smaller the layer size, the faster
        self.weights = [] # the weight for each layer
        self.biases = [] # the bias for each layer

        if self.layer_num == 1:
            # First layer: maps input dimension 1 to output dimension 1
            self.weights.append(nn.Parameter(self.parameter_num, 1))
            self.biases.append(nn.Parameter(1, 1))
        else:
            # First layer: maps input dimension 1 to first hidden layer
            self.weights.append(nn.Parameter(self.parameter_num, self.layer_size))
            self.biases.append(nn.Parameter(self.parameter_num, self.layer_size))
            # Hidden layers
            for _ in range(self.layer_num - 2):
                self.weights.append(nn.Parameter(self.layer_size, self.layer_size))
                self.biases.append(nn.Parameter(1, self.layer_size))
            # Final layer: maps last hidden layer to output dimension 1
            self.weights.append(nn.Parameter(self.layer_size, 1))
            self.biases.append(nn.Parameter(1, 1))

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        prediction = x
        # Forward through hidden layers with ReLU activation
        for i in range(self.layer_num - 1):
            prediction = nn.Linear(prediction, self.weights[i])
            prediction = nn.AddBias(prediction, self.biases[i])
            prediction = nn.ReLU(prediction)
        # Final layer (no activation)
        prediction = nn.Linear(prediction, self.weights[-1])
        prediction = nn.AddBias(prediction, self.biases[-1])
        return prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        def parameters_update(data):
            # get loss
            loss = self.get_loss(*data)
            params = self.weights + self.biases
            grads = nn.gradients(loss, params)
            # update
            for param, grad in zip(params, grads):
                param.update(grad, -self.lr)
            return loss

        for i, data in enumerate(dataset.iterate_forever(self.batch_size)):
            loss = parameters_update(data)
            acc = nn.as_scalar(loss)
            if i % 100 == 0:
                print(f"Epoch {i}: acc={acc}")
            acc_bias = 0.0001  # we can use a better accurancy for stability
            if acc < self.accuracy - acc_bias:
                print(f"Exit with Epoch {i}: acc={acc}")
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.1 # learning rate
        self.batch_size = 100
        self.layer_num = 2 # Number of layers; set to 1 for a simple linear model
        self.layer_size = 256
        self.input_dim = 784
        self.output_dim = 10
        self.accuracy = 0.97 # Threshold for early stopping
        self.weights = []
        self.biases = []

        if self.layer_num == 1:
            # Single layer: directly map input to output
            self.weights.append(nn.Parameter(self.input_dim, self.output_dim))
            self.biases.append(nn.Parameter(1, self.output_dim))
        else:
            # First layer: map input to hidden layer
            self.weights.append(nn.Parameter(self.input_dim, self.layer_size))
            self.biases.append(nn.Parameter(1, self.layer_size))
            # Hidden layers (if any)
            for _ in range(self.layer_num - 2):
                self.weights.append(nn.Parameter(self.layer_size, self.layer_size))
                self.biases.append(nn.Parameter(1, self.layer_size))
            # Final layer: map hidden layer to output
            self.weights.append(nn.Parameter(self.layer_size, self.output_dim))
            self.biases.append(nn.Parameter(1, self.output_dim))

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        logits = x
        if self.layer_num == 1:
            logits = nn.Linear(logits, self.weights[0])
            logits = nn.AddBias(logits, self.biases[0])
        else:
            # Forward propagation through hidden layers with ReLU activation
            for i in range(self.layer_num - 1):
                logits = nn.Linear(logits, self.weights[i])
                logits = nn.AddBias(logits, self.biases[i])
                logits = nn.ReLU(logits)
            # Final layer: no activation
            logits = nn.Linear(logits, self.weights[-1])
            logits = nn.AddBias(logits, self.biases[-1])
        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        def parameters_update(data):
            # Compute loss and gradients
            loss = self.get_loss(*data)
            params = self.weights + self.biases
            grads = nn.gradients(loss, params)
            # Update each parameter using gradient descent
            for param, grad in zip(params, grads):
                param.update(grad, -self.lr)
            return loss

        for i, data in enumerate(dataset.iterate_forever(self.batch_size)):
            loss = parameters_update(data)
            current_loss = nn.as_scalar(loss)
            if i % 100 == 0:
                print(f"Epoch {i}: loss={current_loss}")
            if current_loss < 0.006:
                acc = dataset.get_validation_accuracy()
                acc > self.accuracy
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.accuracy = 0.85
        self.lr = 0.05         # learning rate
        self.batch_size = 100  # batch size
        self.hidden_size = 128 # hidden size

        # W_x: num_chars -> hidden_size
        self.W_x = nn.Parameter(self.num_chars, self.hidden_size)
        # W_h: hidden_size -> hidden_size
        self.W_h = nn.Parameter(self.hidden_size, self.hidden_size)
        # W_out: hidden_size -> len(languages) (output)
        self.W_out = nn.Parameter(self.hidden_size, len(self.languages))
        self.bias_out = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0], self.W_x)
        h = nn.ReLU(h)
        # add x one by one
        for i in range(1, len(xs)):
            x_l = nn.Linear(xs[i], self.W_x)
            h_l = nn.Linear(h, self.W_h)
            z = nn.Add(x_l, h_l)
            h = nn.ReLU(z)
        
        logits = nn.Linear(h, self.W_out)
        logits = nn.AddBias(logits, self.bias_out)
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        def parameters_update(data):
            loss = self.get_loss(*data)
            params = [self.W_x, self.W_h, self.W_out, self.bias_out]
            grads = nn.gradients(loss, params)
            for param, grad in zip(params, grads):
                param.update(grad, -self.lr)
            return loss

        for i, data in enumerate(dataset.iterate_forever(self.batch_size)):
            loss = parameters_update(data)
            if i % 1000 == 0:
                print(f"Epoch {i}: loss = {nn.as_scalar(loss)}")
            if nn.as_scalar(loss) < 0.1:
                acc = dataset.get_validation_accuracy()
                if acc > self.accuracy:
                    print(f"Exit at Epoch {i}: loss = {nn.as_scalar(loss)} and acc = {acc}")
                    break
