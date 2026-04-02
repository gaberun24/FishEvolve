import numpy as np


class NeuralNetwork:
    """Deep recurrent neural network for neuroevolution.

    Supports an optional recurrent connection: the first hidden layer's
    activation is fed back as extra input on the next tick, giving the
    network implicit working memory.
    """

    def __init__(self, topology, recurrent_size=0):
        """topology: list of layer sizes, e.g. [120, 48, 24, 12]
        recurrent_size: number of neurons fed back from hidden1 (0 = feedforward only)
        """
        self.topology = list(topology)
        self.recurrent_size = recurrent_size
        self.weights = []
        self.biases = []
        for i in range(len(topology) - 1):
            w = np.random.randn(topology[i], topology[i + 1]) * 0.5
            b = np.zeros(topology[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        # Recurrent hidden state (zeros initially)
        self.hidden_state = np.zeros(recurrent_size, dtype=np.float64)

        # Stored activations for brain visualization [input, hidden1, hidden2, ..., output]
        self.activations = []

    def feed_forward(self, sensory_inputs):
        """Run sensory inputs through the network.

        If recurrent_size > 0, the previous hidden state is automatically
        concatenated to the sensory inputs before the forward pass.
        Returns the output array.
        """
        x = np.array(sensory_inputs, dtype=np.float64)

        # Prepend recurrent state to form full input
        if self.recurrent_size > 0:
            x = np.concatenate([x, self.hidden_state])

        self.activations = [x.copy()]

        for w, b in zip(self.weights, self.biases):
            x = np.tanh(x @ w + b)
            self.activations.append(x.copy())

        # Feed first hidden layer back as recurrent state for next tick
        if self.recurrent_size > 0 and len(self.activations) > 1:
            self.hidden_state = self.activations[1].copy()

        return x

    def reset_state(self):
        """Reset recurrent hidden state to zeros."""
        self.hidden_state = np.zeros(self.recurrent_size, dtype=np.float64)

    def get_flat_weights(self):
        """Serialize all weights and biases into a single flat array."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts)

    def set_flat_weights(self, flat):
        """Deserialize flat array back into weight matrices and bias vectors."""
        idx = 0
        for i in range(len(self.topology) - 1):
            n_in = self.topology[i]
            n_out = self.topology[i + 1]
            w_size = n_in * n_out
            self.weights[i] = flat[idx:idx + w_size].reshape(n_in, n_out)
            idx += w_size
            self.biases[i] = flat[idx:idx + n_out].copy()
            idx += n_out

    def clone(self):
        nn = NeuralNetwork(self.topology, self.recurrent_size)
        nn.set_flat_weights(self.get_flat_weights())
        return nn

    @staticmethod
    def crossover(a, b):
        """Uniform crossover: for each weight, pick from parent a or b."""
        child = NeuralNetwork(a.topology, a.recurrent_size)
        flat_a = a.get_flat_weights()
        flat_b = b.get_flat_weights()
        mask = np.random.random(len(flat_a)) < 0.5
        flat_child = np.where(mask, flat_a, flat_b)
        child.set_flat_weights(flat_child)
        return child

    @staticmethod
    def mutate(nn, rate=0.1, strength=0.3):
        """Mutate weights in-place."""
        flat = nn.get_flat_weights()
        mutations = np.random.random(len(flat)) < rate
        flat[mutations] += np.random.randn(mutations.sum()) * strength
        # Occasional full reset of a weight (1% chance per weight)
        resets = np.random.random(len(flat)) < 0.01
        flat[resets] = np.random.randn(resets.sum())
        nn.set_flat_weights(flat)
