import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.numTrainingGames = 2000
        self.batch_size = 256
        self.layer_size = 256
        self.layer_num = 2  # number of layers (set to 1 for a simple linear model)
        
        # Store parameters in the order of the forward pass.
        self.parameters = []
        self.weights = []
        self.biases = []
        
        if self.layer_num == 1:
            # Single layer network: map input directly to output
            w = nn.Parameter(self.state_size, self.num_actions)
            b = nn.Parameter(1, self.num_actions)
            self.weights.append(w)
            self.biases.append(b)
            self.parameters.extend([w, b])
        else:
            # First layer: input to hidden layer
            w = nn.Parameter(self.state_size, self.layer_size)
            b = nn.Parameter(1, self.layer_size)
            self.weights.append(w)
            self.biases.append(b)
            self.parameters.extend([w, b])
            
            # Hidden layers (if any)
            for _ in range(self.layer_num - 2):
                w = nn.Parameter(self.layer_size, self.layer_size)
                b = nn.Parameter(1, self.layer_size)
                self.weights.append(w)
                self.biases.append(b)
                self.parameters.extend([w, b])
            
            # Final layer: hidden to output mapping
            w = nn.Parameter(self.layer_size, self.num_actions)
            b = nn.Parameter(1, self.num_actions)
            self.weights.append(w)
            self.biases.append(b)
            self.parameters.extend([w, b])

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        logits = states
        if self.layer_num == 1:
            # Single layer: linear transform
            logits = nn.Linear(logits, self.weights[0])
            logits = nn.AddBias(logits, self.biases[0])
        else:
            # Hidden layers with ReLU activation
            # Process all layers except final layer.
            for i in range(self.layer_num - 1):
                logits = nn.Linear(logits, self.weights[i])
                logits = nn.AddBias(logits, self.biases[i])
                logits = nn.ReLU(logits)
            # Final layer: no activation
            logits = nn.Linear(logits, self.weights[-1])
            logits = nn.AddBias(logits, self.biases[-1])
        return logits

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        grads = nn.gradients(loss, self.parameters)

        for param, grad in zip(self.parameters, grads):
            param.update(grad, -self.learning_rate)
