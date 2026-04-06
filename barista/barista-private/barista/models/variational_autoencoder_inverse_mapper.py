import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from barista.models import conditional_neural_process_model
from barista.models.conditional_neural_process_model import CNPDeterministicEncoder
from barista.models.conditional_neural_process_model import CNPDeterministicDecoder


class VAIM(nn.Module):
    def __init__(self, forward_encoder_sizes, forward_decoder_sizes, backward_decoder_sizes, is_binary=True):
        super(VAIM, self).__init__()

        self.is_binary = is_binary
        self.forward_mapper = CNPForwardMapper(forward_encoder_sizes, forward_decoder_sizes) #encoder
        self.backward_mapper = VAIMBackwardMapper(backward_decoder_sizes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, context_x, context_y, target_x, target_y=None):
        # Forward mapper
        (y_rec, mu, logvar) = self.forward_mapper(((context_x, context_y), target_x), is_binary=self.is_binary)
        z = self.reparameterize(mu, logvar)
        x_rec = self.backward_mapper(z, y_rec)

        return (x_rec, y_rec)

class VAIMBackwardMapper(nn.Module):
    def __init__(self, output_sizes, leaky_relu_slope=0.01):
        super(VAIMBackwardMapper, self).__init__()
        self.linears = nn.ModuleList()
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)

        for i in range(len(output_sizes) - 1):
            self.linears.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))

    def forward(self, z_latent, y_rec):
        batch_size, num_total_points, _ = y_rec.shape

        x = torch.cat((z_latent, y_rec), dim=-1)
        x = x.view(batch_size * num_total_points, -1)

        h = x
        residuals = [None] * len(self.linears)

        for i, layer in enumerate(self.linears[:-1]):
            h = layer(h)
            if i >= 2:
                h = h + residuals[i - 2]
            h = self.activation(h)
            residuals[i] = h  # store for next residual

        # Final layer without activation
        h = self.linears[-1](h)

        return h.view(batch_size, num_total_points, -1)

class CNPForwardDecoder(nn.Module):
    def __init__(self, output_sizes):
        """CNP decoder.
        Args:
            output_sizes: An iterable containing the output sizes of the decoder MLP.
        """
        super(CNPForwardDecoder, self).__init__()
        self.output_sizes = output_sizes
        self.linears = nn.ModuleList()
        for i in range(len(output_sizes) - 2):
            self.linears.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))

        # Replace the last layer with the combined output layer
        final_input_dim = output_sizes[-3]
        final_output_dim = output_sizes[-2] + 2 * output_sizes[-1]
        self.linears.append(nn.Linear(final_input_dim, final_output_dim))

    def forward(self, representation, target_x, is_binary):
        """Decodes the individual targets.

        Args:
            representation: The encoded representation of the context
            target_x: The x locations for the target query

        Returns:
            dist: A multivariate Gaussian over the target points.
            mu: The mean of the multivariate Gaussian.
            sigma: The standard deviation of the multivariate Gaussian.   
        """

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_total_points, _ = target_x.shape
        representation = representation.unsqueeze(1).repeat([1, num_total_points, 1])

        # Concatenate the representation and the target_x
        input = torch.cat((representation, target_x), dim=-1)
        hidden = input.view(batch_size * num_total_points, -1)

        # Pass through MLP
        for i, linear in enumerate(self.linears[:-1]):
            hidden = torch.relu(linear(hidden))

        # Final layer (no activation)
        hidden = self.linears[-1](hidden)

        # Reshape back to original
        hidden = hidden.view(batch_size, num_total_points, -1)

        # Split into outputs
        y_rec, mu, sigma = torch.split(hidden, [self.output_sizes[-2], self.output_sizes[-1], self.output_sizes[-1]], dim=-1)

        # Map mu to a value between 0 and 1 and get the expectation and variance
        if is_binary==True:
            y_rec = torch.sigmoid(y_rec)
            #mu, sigma = self.sigmoid_expectation(mu, sigma)

        # Get the distribution
        # Ensure scale is strictly positive before passing to Normal distribution
        #sigma = torch.clamp(sigma, min=1e-4)  # OR
        #sigma = F.softplus(sigma) + 1e-6
        return y_rec, mu, sigma
    
    def sigmoid_expectation(self, mu, sigma):
        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(sigma)

        # Calculate y = sqrt(1 + 3 / pi^2 * sigma^2)
        y = torch.sqrt(1. + 3. / (np.pi ** 2) * sigma ** 2)

        # Prevent division by zero
        y = torch.clamp(y, min=1e-4)

        # Calculate expectation and variance
        expectation = torch.sigmoid(mu / y)
        var = expectation * (1. - expectation) * (1. - (1. / y))

        return expectation, var

class CNPForwardMapper(nn.Module):
    def __init__(self, forward_encoder_sizes, forward_decoder_sizes):
        super(CNPForwardMapper, self).__init__()
        """Initialises the model.

        Args:
            encoder_output_sizes: An iterable containing the sizes of hidden layers of
                the encoder. The last one is the size of the representation r.
            decoder_output_sizes: An iterable containing the sizes of hidden layers of
                the decoder. The last element should correspond to the dimension of
                the y * 2 + y (it encodes both mean and variance concatenated and a reconstruction of y)
        """
        self._encoder = CNPDeterministicEncoder(forward_encoder_sizes)
        self._decoder = CNPDeterministicDecoder(forward_decoder_sizes)

    def forward(self, query, target_y=None, is_binary=True):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                context_x: Array of shape batch_size x num_context x 1 contains the 
                    x values of the context points.
                context_y: Array of shape batch_size x num_context x 1 contains the 
                    y values of the context points.
                target_x: Array of shape batch_size x num_target x 1 contains the
                    x values of the target points.
            target_y: The ground truth y values of the target y. An array of 
                shape batchsize x num_targets x 1.

        Returns:
            log_p: The log_probability of the target_y given the predicted
            distribution.
            mu: The mean of the predicted distribution.
            sigma: The variance of the predicted distribution.
        """

        (context_x, context_y), target_x = query
        # Pass query through the encoder and the decoder

        representation = self._encoder(context_x, context_y)
        y_rec, mu, sigma = self._decoder(representation, target_x, is_binary)
        
        return y_rec, mu, sigma