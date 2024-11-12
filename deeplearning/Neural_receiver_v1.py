import torch
import torch.nn as nn
import numpy as np


class NeuralReceiver(nn.Module):
    def __init__(self):
        """
        Neural Receiver that processes received signals to remove noise and channel effects.
        """
        super(NeuralReceiver, self).__init__()

        # Calculate input dimensions based on y shape (128, 1, 1, 14, 76)
        self.input_features = 14 * 76  # 1064

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate the size after convolutions
        self.conv_output_size = 128 * 14 * 76  # Maintaining spatial dimensions

        # Dense layers for signal processing
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 768),  # Output dimension matches required x shape
        )

    def forward(self, y, h_b, tau_b, h_out, no):
        """
        Forward pass of the neural receiver.

        Args:
            y (torch.Tensor): Received signal (128, 1, 1, 14, 76)
            h_b (torch.Tensor): Channel impulse response (128, 1, 1, 1, 16, 10, 1)
            tau_b (torch.Tensor): Delay profile (128, 1, 1, 10)
            h_out (torch.Tensor): Channel response (128, 1, 1, 1, 16, 1, 76)
            no (float): Noise power

        Returns:
            tuple: (x_hat, no_eff)
                - x_hat (torch.Tensor): Processed signal (128, 1, 1, 768)
                - no_eff (float): Effective noise power
        """
        batch_size = y.shape[0]

        # Reshape input for 2D convolution: (batch, channels, height, width)
        y = y.view(batch_size, 1, 14, 76)

        # Apply convolution layers
        x = self.conv_layers(y)

        # Flatten for dense layers
        x = x.view(batch_size, -1)

        # Apply dense layers
        x = self.fc_layers(x)

        # Reshape output to match required dimensions (128, 1, 1, 768)
        x_hat = x.view(batch_size, 1, 1, -1)

        # Calculate effective noise power
        no_eff = self._calculate_effective_noise(x_hat, h_out, no)

        return x_hat, no_eff

    def _calculate_effective_noise(self, x_hat, h_out, no):
        """
        Calculate effective noise power based on the processed signal and channel response.

        Args:
            x_hat (torch.Tensor): Processed signal
            h_out (torch.Tensor): Channel response
            no (float): Original noise power

        Returns:
            float: Effective noise power
        """
        signal_power = torch.mean(torch.abs(x_hat) ** 2)
        channel_effect = torch.mean(torch.abs(h_out) ** 2)
        no_eff = no * signal_power * channel_effect
        return float(no_eff.item())


def numpy_to_torch(arr):
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(arr).float()


def process_signal(y, x, no, h_b, tau_b, h_out):
    """
    Process a signal using the Neural Receiver.

    Args:
        y (numpy.ndarray): Received signal (128, 1, 1, 14, 76)
        x (numpy.ndarray): Transmitted signal (128, 1, 1, 768)
        no (float): Noise power
        h_b (numpy.ndarray): Channel impulse response (128, 1, 1, 1, 16, 10, 1)
        tau_b (numpy.ndarray): Delay profile (128, 1, 1, 10)
        h_out (numpy.ndarray): Channel response (128, 1, 1, 1, 16, 1, 76)

    Returns:
        tuple: (x_hat, no_eff)
            - x_hat (numpy.ndarray): Processed signal (128, 1, 1, 768)
            - no_eff (float): Effective noise power
    """
    # Convert numpy arrays to torch tensors
    y_torch = numpy_to_torch(y)
    h_b_torch = numpy_to_torch(h_b)
    tau_b_torch = numpy_to_torch(tau_b)
    h_out_torch = numpy_to_torch(h_out)

    # Initialize and use the receiver
    receiver = NeuralReceiver()
    x_hat, no_eff = receiver(y_torch, h_b_torch, tau_b_torch, h_out_torch, no)

    # Convert output back to numpy
    x_hat_np = x_hat.detach().numpy()

    return x_hat_np, no_eff


# class NeuralReceiverModel(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralReceiverModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(
#             64, 6
#         )  # Output for x_hat, no_eff, h_hat, err_var, h_perfect, err_var_perfect

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x


# class NeuralReceiver:
#     def __init__(self, input_size=1064):  # Set the input size to match your data
#         self.model = NeuralReceiverModel(input_size)

#     def neural_receiver_method(
#         self, y, no, h_b=None, tau_b=None, h_out=None, perfect_csi=False
#     ):
#         # Convert received signal to tensor
#         y_tensor = torch.tensor(y, dtype=torch.float32)

#         # Perform inference with the neural receiver
#         with torch.no_grad():  # Disable gradient calculation for inference
#             output = self.model(y_tensor)

#         print("output:", output.shape)  # Check the output shape

#         # Check if output is a single tensor with 6 elements
#         if output.dim() == 1 and output.shape[0] == 6:
#             # If the model is returning a single output, handle accordingly
#             x_hat = output[:2]  # This will give you the first two elements
#             no_eff = output[2].item()  # Assuming the third element is no_eff
#         else:
#             # Split the output into respective variables
#             x_hat = output[0]  # Estimated transmitted signal
#             no_eff = output[1]  # Effective noise

#             # Convert outputs to numpy arrays for compatibility
#             x_hat = x_hat.numpy()  # Convert to numpy array

#             # Check if no_eff is a tensor with multiple elements
#             if no_eff.dim() > 0 and no_eff.shape[0] > 1:
#                 no_eff = (
#                     no_eff.mean().item()
#                 )  # Take the mean if it's a tensor with multiple elements
#             else:
#                 no_eff = no_eff.item()  # Convert to float if it's a single value

#         # Reshape x_hat to the required shape (128, 2, 2, 768) if applicable
#         if x_hat.shape[0] == 6:
#             # If x_hat is of shape (6,), we need to determine how to process it
#             x_hat = np.zeros(
#                 (128, 2, 2, 768)
#             )  # Create a dummy output with the desired shape
#         else:
#             # If x_hat is not of the expected shape, handle accordingly
#             raise ValueError(f"Unexpected shape for x_hat: {x_hat.shape}")

#         return x_hat, no_eff  # Return only x_hat and no_eff
