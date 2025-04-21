import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# setting seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ConvConceptizer class
class ConvConceptizer(nn.Module):
    def __init__(self, image_size, num_concepts, concept_dim, image_channels=1, encoder_channels=(10,),
                 decoder_channels=(16, 8), kernel_size_conv=5, kernel_size_upsample=(5, 5, 2),
                 stride_conv=1, stride_pool=2, stride_upsample=(2, 1, 2),
                 padding_conv=0, padding_upsample=(0, 0, 1), **kwargs):
        """
        CNN Autoencoder used to learn the concepts, present in an input image
        description:
            The ConvConceptizer is a convolutional autoencoder that learns a set of concepts from an input image.
            It consists of an encoder and a decoder.
            The encoder takes an image as input and learns a set of concepts.
            The decoder takes the concepts as input and reconstructs the image.

        Parameters
        ----------
        image_size : int
            the width of the input image
        num_concepts : int
            the number of concepts
        concept_dim : int
            the dimension of each concept to be learned
        image_channels : int
            the number of channels of the input images
        encoder_channels : tuple[int]
            a list with the number of channels for the hidden convolutional layers
        decoder_channels : tuple[int]
            a list with the number of channels for the hidden upsampling layers
        kernel_size_conv : int, tuple[int]
            the size of the kernels to be used for convolution
        kernel_size_upsample : int, tuple[int]
            the size of the kernels to be used for upsampling
        stride_conv : int, tuple[int]
            the stride of the convolutional layers
        stride_pool : int, tuple[int]
            the stride of the pooling layers
        stride_upsample : int, tuple[int]
            the stride of the upsampling layers
        padding_conv : int, tuple[int]
            the padding to be used by the convolutional layers
        padding_upsample : int, tuple[int]
            the padding to be used by the upsampling layers
        """
        super(ConvConceptizer, self).__init__()
        self.num_concepts = num_concepts
        self.filter = filter
        self.dout = image_size

        # Encoder params
        encoder_channels = (image_channels,) + encoder_channels
        kernel_size_conv = handle_integer_input(kernel_size_conv, len(encoder_channels))
        stride_conv = handle_integer_input(stride_conv, len(encoder_channels))
        stride_pool = handle_integer_input(stride_pool, len(encoder_channels))
        padding_conv = handle_integer_input(padding_conv, len(encoder_channels))
        encoder_channels += (num_concepts,)

        # Decoder params
        decoder_channels = (num_concepts,) + decoder_channels
        kernel_size_upsample = handle_integer_input(kernel_size_upsample, len(decoder_channels))
        stride_upsample = handle_integer_input(stride_upsample, len(decoder_channels))
        padding_upsample = handle_integer_input(padding_upsample, len(decoder_channels))
        decoder_channels += (image_channels,)

        # Encoder implementation
        """
        Creates sequence of convolutional layers. Uses conv_block function to create each block.
        Each block includes:
            Convolution (Conv2d)
            Max Pooling
            ReLU activation
        Reduces image dimensions while extracting features
        Flow : 
        Encoding Process: Image → Convolutions → Concepts
        Decoding Process: Concepts → Transposed Convolutions → Reconstructed Image
        Concept Learning: Forces network to learn meaningful, interpretable features
        """
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(self.conv_block(in_channels=encoder_channels[i],
                                                out_channels=encoder_channels[i + 1],
                                                kernel_size=kernel_size_conv[i],
                                                stride_conv=stride_conv[i],
                                                stride_pool=stride_pool[i],
                                                padding=padding_conv[i]))
            self.dout = (self.dout - kernel_size_conv[i] + 2 * padding_conv[i] + stride_conv[i] * stride_pool[i]) // (
                    stride_conv[i] * stride_pool[i])

        # if self.filter and concept_dim == 1:
        self.encoder.append(ScalarMapping((self.num_concepts, self.dout, self.dout)))
        # else:
        #     self.encoder.append(Flatten())
        #     self.encoder.append(nn.Linear(self.dout ** 2, concept_dim))

        # Decoder implementation
        """
        Mirror of encoder
        Uses transposed convolutions (ConvTranspose2d) for upsampling
        Reconstructs original image from concepts
        """
        self.unlinear = nn.Linear(concept_dim, self.dout ** 2)
        self.decoder = nn.ModuleList()
        decoder = []
        for i in range(len(decoder_channels) - 1):
            decoder.append(self.upsample_block(in_channels=decoder_channels[i],
                                               out_channels=decoder_channels[i + 1],
                                               kernel_size=kernel_size_upsample[i],
                                               stride_deconv=stride_upsample[i],
                                               padding=padding_upsample[i]))
            decoder.append(nn.ReLU(inplace=True))
        decoder.pop()
        decoder.append(nn.Tanh())
        self.decoder = nn.ModuleList(decoder)

    def forward(self, x):
        """
        Forward pass of the general conceptizer.

        Computes concepts present in the input.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        encoded : torch.Tensor
            Encoded concepts (batch_size, concept_number, concept_dimension)
        decoded : torch.Tensor
            Reconstructed input (batch_size, *)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

    def encode(self, x):
        """
        The encoder part of the autoencoder which takes an Image as an input
        and learns its hidden representations (concepts)

        Parameters
        ----------
        x : Image (batch_size, channels, width, height)

        Returns
        -------
        encoded : torch.Tensor (batch_size, concept_number, concept_dimension)
            the concepts representing an image

        """
        encoded = x
        for module in self.encoder:
            encoded = module(encoded)
        return encoded

    def decode(self, z):
        """
        The decoder part of the autoencoder which takes a hidden representation as an input
        and tries to reconstruct the original image

        Parameters
        ----------
        z : torch.Tensor (batch_size, channels, width, height)
            the concepts in an image

        Returns
        -------
        reconst : torch.Tensor (batch_size, channels, width, height)
            the reconstructed image

        """
        reconst = self.unlinear(z)
        reconst = reconst.view(-1, self.num_concepts, self.dout, self.dout)
        for module in self.decoder:
            reconst = module(reconst)
        return reconst

    def conv_block(self, in_channels, out_channels, kernel_size, stride_conv, stride_pool, padding):
        """
        A helper function that constructs a convolution block with pooling and activation
        Creates encoder convolution blocks

        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_conv : int
            the stride of the deconvolution
        stride_pool : int
            the stride of the pooling layer
        padding : int
            the size of padding

        Returns
        -------
        sequence : nn.Sequence
            a sequence of convolutional, pooling and activation modules
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride_conv,
                      padding=padding),
            # nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=stride_pool,
                         padding=padding),
            nn.ReLU(inplace=True)
        )

    def upsample_block(self, in_channels, out_channels, kernel_size, stride_deconv, padding):
        """
        A helper function that constructs an upsampling block with activations
        Creates decoder upsampling blocks

        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_deconv : int
            the stride of the deconvolution
        padding : int
            the size of padding

        Returns
        -------
        sequence : nn.Sequence
            a sequence of deconvolutional and activation modules
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride_deconv,
                               padding=padding),
        )


class Flatten(nn.Module):
    def forward(self, x):
        """
        Flattens the inputs to only 3 dimensions, preserving the sizes of the 1st and 2nd.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (dim1, dim2, *).

        Returns
        -------
        flattened : torch.Tensor
            Flattened input (dim1, dim2, dim3)
        """
        return x.view(x.size(0), x.size(1), -1)


def handle_integer_input(input, desired_len):
    """
    Useful for layer-wise parameters (kernel sizes, strides, padding)

    Checks if the input is an integer or a list.
    If an integer, it is replicated the number of  desired times
    If a tuple, the tuple is returned as it is

    Parameters
    ----------
    input : int, tuple
        The input can be either a tuple of parameters or a single parameter to be replicated
    desired_len : int
        The length of the desired list

    Returns
    -------
    input : tuple[int]
        a tuple of parameters which has the proper length.
    """
    if type(input) is int:
        return (input,) * desired_len
    elif type(input) is tuple:
        if len(input) != desired_len:
            raise AssertionError("The sizes of the parameters for the CNN conceptizer do not match."
                                 f"Expected '{desired_len}', but got '{len(input)}'")
        else:
            return input
    else:
        raise TypeError(f"Wrong type of the parameters. Expected tuple or int but got '{type(input)}'")


class ScalarMapping(nn.Module):
    def __init__(self, conv_block_size):
        """
        Module that maps each filter of a convolutional block to a scalar value
        dimensionality reduction for convolutional outputs.

        Parameters
        ----------
        conv_block_size : tuple (int iterable)
            Specifies the size of the input convolutional block: (NUM_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH)
        """
        super().__init__()
        self.num_filters, self.filter_height, self.filter_width = conv_block_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_filters):
            self.layers.append(nn.Linear(self.filter_height * self.filter_width, 1))

    def forward(self, x):
        """
        Reduces a 3D convolutional block to a 1D vector by mapping each 2D filter to a scalar value.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH).

        Returns
        -------
        mapped : torch.Tensor
            Reduced input (BATCH, CHANNELS, 1)
        """
        x = x.view(-1, self.num_filters, self.filter_height * self.filter_width)
        mappings = []
        for f, layer in enumerate(self.layers):
            mappings.append(layer(x[:, [f], :]))
        return torch.cat(mappings, dim=1)

class ConvParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, cl_sizes=(1, 10, 20), kernel_size=5, hidden_sizes=None, dropout=0.5,
                 **kwargs):
        """Parameterizer for MNIST dataset.

        Consists of convolutional as well as fully connected modules.
        Determines the relevance/importance of learned concepts for each class in MNIST classification. 
        Input Image (BATCH, channels, 28, 28)
        ↓ Convolutional Processing
        Feature Maps
        ↓ Flatten
        1D Feature Vector
        ↓ Fully Connected Layers
        Relevance Scores (BATCH, NUM_CONCEPTS, NUM_CLASSES)


        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        cl_sizes : iterable of int
            Indicates the number of kernels of each convolutional layer in the network. The first element corresponds to
            the number of input channels.
        kernel_size : int
            Indicates the size of the kernel window for the convolutional layers.
        hidden_sizes : iterable of int or None
            Indicates the size of each fully connected layer in the network. If None, it will be calculated based on the
            convolutional output size. The last element must be equal to the number of concepts multiplied with the
            number of output classes.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.cl_sizes = cl_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build convolutional layers
        cl_layers = []
        for h, h_next in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(nn.Conv2d(h, h_next, kernel_size=self.kernel_size))
            cl_layers.append(nn.MaxPool2d(2, stride=2))
            cl_layers.append(nn.ReLU())
        # dropout before maxpool
        cl_layers.insert(-2, nn.Dropout2d(self.dropout))
        self.cl_layers = nn.Sequential(*cl_layers)

        # Calculate the size of the flattened convolutional output
        # For MNIST (28x28), after 2 convolutional layers with kernel_size=5 and 2 max pooling layers with stride=2,
        # the feature map size will be ((28-5+1)//2-5+1)//2 = 4, and with 20 channels, this gives us 4*4*20 = 320 features
        # This calculation depends on the exact network architecture and input size
        with torch.no_grad():
            # Create a dummy input to get the shape
            dummy_input = torch.zeros(1, cl_sizes[0], 28, 28)  # Assuming MNIST 28x28
            conv_output = self.cl_layers(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)

        # If hidden_sizes is not provided, create it
        if hidden_sizes is None:
            hidden_sizes = [flattened_size, 100, 50, num_concepts * num_classes]
        else:
            # Make sure the first element matches the flattened size
            hidden_sizes = list(hidden_sizes)
            hidden_sizes[0] = flattened_size

        self.hidden_sizes = hidden_sizes

        # Build fully connected layers
        fc_layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            fc_layers.append(nn.Linear(h, h_next))
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.ReLU())
        fc_layers.pop()  # Remove the last ReLU
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass of MNIST parameterizer.

        Computes relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        return self.fc_layers(flattened).view(-1, self.num_concepts, self.num_classes)

#losses

def mnist_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data
    jacobian of predictions (J_yx) wrt x tells us how much each pixel in the image affects the prediction
    jacobian of concepts (J_hx) wrt x tells us how much each concept affects the prediction
    robustness loss is the difference between the jacobian of predictions wrt x and the jacobian of concepts wrt x
    """
    # Ensure input has requires_grad
    if not x.requires_grad:
        x.requires_grad_()

    # concept_dim is always 1
    concepts = concepts.squeeze(-1)
    aggregates = aggregates.squeeze(-1)

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x / jacobian of predictions wrt x
    jacobians = []
    for i in range(num_classes):
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:, i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x,
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:, i] = 1.
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x,
                                   grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size, -1).unsqueeze(-1))
    # bs x num_features x num_concepts
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    robustness_loss = J_yx - torch.bmm(J_hx, relevances)

    return robustness_loss.norm(p='fro')

class SumAggregator(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator.

        Aggregates concepts and relevances and returns the predictions for each class.

        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS, NUM_CLASSES).

        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class. Shape - (BATCH, NUM_CLASSES)

        """
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SENN_CF(nn.Module):
    def __init__(self, conceptizer, parameterizer, aggregator):
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):

        """Forward pass of SENN module.
        Input Image
        ↓
        Concepts + Reconstruction (Conceptizer)
        ↓
        Relevance Scores (Parameterizer)
        ↓
        Final Prediction (Aggregator)
        ↓
        Returns: (predictions, (concepts, relevances), reconstructed_image)"""
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x

    def get_counterfactual_explanations(self, x, orig_class=None, steps=100, lr=0.01, reg_lambda=0.1, concept_lambda=1.0):
        """
        Generate counterfactual explanations for all classes using SENN's conceptual understanding
        """
        # Get original prediction if not provided
        with torch.no_grad():
            pred_orig, (concepts_orig, relevances_orig), _ = self(x)
            orig_class = orig_class or pred_orig.argmax(1).item()

        results = []
        # Generate counterfactuals for all classes except original
        for target_class in range(self.aggregator.num_classes):
            if target_class == orig_class:
                continue

            cf_result = self.get_concept_based_counterfactual(
                x,
                orig_class,
                target_class,
                concepts_orig,
                relevances_orig,
                steps=steps,
                lr=lr,
                reg_lambda=reg_lambda,
                concept_lambda=concept_lambda
            )
            results.append((target_class, cf_result))

        return orig_class, results

    def get_concept_based_counterfactual(self, x, orig_class, target_class, concepts_orig, relevances_orig,
                                       steps=100, lr=0.01, reg_lambda=0.1, concept_lambda=1.0):
        """
        Generate counterfactual using SENN's conceptual understanding
        description:
            Description:
            1. Initialization:
            - Starts with x_cf = x.clone() - a copy of the original image
            - Sets up gradient tracking and Adam optimizer

            2. Loss Components for Optimization:
            - Classification Loss: Push towards target class prediction
            - Concept Loss: 
                * Encourages target class concepts 
                * Discourages original class concepts
                * Example: If we want to change the digit from 3 to 5, we want to encourage the concept of 5(class concept) and discourage the concept of 3(original class concept)
            - Proximity Loss: Keeps changes minimal (torch.norm(x_cf - x))

            3. Optimization Process:
            - Makes gradual changes through optimization steps
            - Clips values to valid image range (0,1)
            - Tracks best result based on total loss

            4. Early Stopping & Best Result:
            - Stops if target class achieved with minimal loss
            - Maintains best counterfactual and metrics throughout

            5. Control Parameters:
            - steps: Number of optimization iterations
            - lr: Learning rate for changes
            - reg_lambda: Weight for proximity loss
            - concept_lambda: Weight for concept alignment

            6. Returns:
            - Best counterfactual image
            - Comprehensive metrics (losses, concepts, relevances)
        """
        # Create a copy that requires gradients
        x_cf = x.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x_cf], lr=lr)

        # Get target concepts we want to achieve
        target_relevances = relevances_orig[0, :, target_class]
        orig_relevances = relevances_orig[0, :, orig_class]

        # Track best result
        best_cf = None
        best_loss = float('inf')
        best_metrics = None

        for step in range(steps):
            optimizer.zero_grad()

            # Get current counterfactual predictions and concepts
            pred_cf, (concepts_cf, relevances_cf), _ = self(x_cf)

            # 1. Classification Loss - push toward target class
            class_loss = -F.log_softmax(pred_cf, dim=1)[0, target_class]

            # 2. Concept Alignment Loss
            # Encourage important target concepts
            target_concept_loss = -torch.sum(
                torch.abs(target_relevances) * concepts_cf[0, :, 0]
            )

            # Discourage original class concepts
            orig_concept_loss = torch.sum(
                torch.abs(orig_relevances) * concepts_cf[0, :, 0]
            )

            concept_loss = concept_lambda * (target_concept_loss + orig_concept_loss)

            # 3. Proximity Loss - keep changes minimal
            proximity_loss = reg_lambda * torch.norm(x_cf - x)

            # Total loss
            total_loss = class_loss + concept_loss + proximity_loss

            # Optimization step
            total_loss.backward()
            optimizer.step()

            # Clip to valid image range
            with torch.no_grad():
                x_cf.data.clamp_(0, 1)

                # Check current prediction
                pred_current, (concepts_current, relevances_current), _ = self(x_cf)
                current_class = pred_current.argmax(1).item()

                # Update best if improved
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_cf = x_cf.clone()
                    best_metrics = {
                        'concepts_orig': concepts_orig.detach(),
                        'concepts_cf': concepts_current.detach(),
                        'relevances_orig': relevances_orig.detach(),
                        'relevances_cf': relevances_current.detach(),
                        'class_loss': class_loss.item(),
                        'concept_loss': concept_loss.item(),
                        'proximity_loss': proximity_loss.item(),
                        'total_loss': total_loss.item()
                    }

                # Early stopping if we found a good counterfactual
                if current_class == target_class and total_loss.item() < best_loss:
                    break

        return best_cf.detach(), best_metrics

    def visualize_counterfactual_explanation(self, x, target_class, cf_result):
        """
        Visualize counterfactual with concept relevance comparison
        """
        counterfactual, metrics = cf_result

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Original image
        axes[0,0].imshow(x[0,0].cpu().numpy(), cmap='gray')
        axes[0,0].set_title('Original Image')

        # Counterfactual image
        axes[0,1].imshow(counterfactual[0,0].cpu().detach().numpy(), cmap='gray')
        axes[0,1].set_title(f'Counterfactual (Class {target_class})')

        # Original concept relevances
        concepts_orig = metrics['concepts_orig'][0,:,0].cpu().numpy()
        relevances_orig = metrics['relevances_orig'][0,:,target_class].cpu().numpy()
        axes[1,0].bar(range(len(concepts_orig)), concepts_orig * relevances_orig)
        axes[1,0].set_title('Original Concept Relevances')

        # Counterfactual concept relevances
        concepts_cf = metrics['concepts_cf'][0,:,0].cpu().numpy()
        relevances_cf = metrics['relevances_cf'][0,:,target_class].cpu().numpy()
        axes[1,1].bar(range(len(concepts_cf)), concepts_cf * relevances_cf)
        axes[1,1].set_title('Counterfactual Concept Relevances')

        plt.tight_layout()
        return fig

# 2. Data Loading
def load_mnist_data(batch_size=64):
    """Load MNIST dataset into train and test loaders"""
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_senn(model, train_loader, num_epochs, learning_rate=0.001, rob_lambda=0.1):
    """Train the SENN model

    loss function: NLL
    optimizer: Adam

    For each epoch:
    For each batch:
        1. Forward Pass → Get predictions & concepts
        2. Calculate Classification Loss
        3. Calculate Robustness Loss
        4. Combine Losses
        5. Backpropagate & Update
        6. Track & Report Progress"""
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure inputs require gradients for robustness loss
            inputs.requires_grad_(True)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            preds, explanations, recon_x = model(inputs)
            concepts, relevances = explanations

            # Calculate loss
            task_loss = criterion(preds, labels)

            # Calculate robustness loss - key for interpretability
            rob_loss = mnist_robustness_loss(inputs, preds.unsqueeze(-1), concepts, relevances)

            # Total loss: task loss + robustness loss
            total_loss = task_loss + rob_lambda * rob_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0

    print('Training complete')

# 4. Model Evaluation
def evaluate_senn(model, test_loader):
    """Evaluate SENN model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            preds, _, _ = model(inputs)

            # Get predictions
            _, predicted = torch.max(preds.data, 1)

            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'SENN model classification test accuracy: {accuracy:.2f}%')
    return accuracy

# 5. Counterfactual Generation and Evaluation
def generate_evaluate_counterfactuals(model, test_loader, num_samples=100, target_strategy='next'):
    """
    Generate and evaluate counterfactuals for a set of test samples

    Evaluation Focus:
    - Calculates quantitative metrics (L1/L2 distances, sparsity, success rate)
    - Tracks statistical measures across many samples
    - Good for model performance evaluation

    Args:
        model: Trained SENN model
        test_loader: Test data loader
        num_samples: Number of samples to generate counterfactuals for
        target_strategy: Strategy to select target class ('next', 'random')
    """
    model.eval()

    # Statistics to track
    success_count = 0
    total_l2_dist = 0
    total_l1_dist = 0
    sparsity_scores = []
    concept_changes = []

    # Get samples
    test_iter = iter(test_loader)
    samples_processed = 0

    while samples_processed < num_samples:
        try:
            images, labels = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            images, labels = next(test_iter)

        batch_size = min(images.size(0), num_samples - samples_processed)
        images = images[:batch_size].to(device)
        labels = labels[:batch_size].to(device)

        # Process each image
        for i in range(batch_size):
            image = images[i:i+1]
            label = labels[i].item()

            # Determine target class
            if target_strategy == 'next':
                target_class = (label + 1) % 10
            elif target_strategy == 'random':
                target_class = torch.randint(0, 10, (1,)).item()
                while target_class == label:
                    target_class = torch.randint(0, 10, (1,)).item()

            # Generate counterfactual
            orig_class, counterfactuals = model.get_counterfactual_explanations(
                image, 
                orig_class=label,
                steps=100,#------------------------------------change to 100 for normal run
                lr=0.01,
                reg_lambda=0.1,
                concept_lambda=1.0
            )

            # Find the counterfactual for our target class
            for cf_class, (cf, metrics) in counterfactuals:
                if cf_class == target_class:
                    # Calculate metrics
                    l2_dist = torch.norm(cf - image).item()
                    l1_dist = torch.norm(cf - image, p=1).item()
                    diff = torch.abs(cf - image)
                    sparsity = (diff > 0.1).float().mean().item()
                    
                    total_l2_dist += l2_dist
                    total_l1_dist += l1_dist
                    sparsity_scores.append(sparsity)
                    concept_changes.append(metrics['concept_loss'])
                    success_count += 1

                    # Occasionally visualize
                    if samples_processed % 10 == 0:
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(image[0, 0].cpu().numpy(), cmap='gray')
                        plt.title(f'Original (Class {label})')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(cf[0, 0].cpu().numpy(), cmap='gray')
                        plt.title(f'Counterfactual (Class {target_class})')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(diff[0, 0].cpu().numpy(), cmap='hot')
                        plt.colorbar()
                        plt.title('Changes Made')
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.show()
                    break

            samples_processed += 1
            if samples_processed >= num_samples:
                break

    # Print overall statistics
    print(f"\nCounterfactual Evaluation Results ({num_samples} samples):")
    print(f"Success Rate: {success_count/num_samples:.4f}")

    if success_count > 0:
        print(f"Average L2 Distance: {total_l2_dist/success_count:.4f}")
        print(f"Average L1 Distance: {total_l1_dist/success_count:.4f}")
        print(f"Average Sparsity (% pixels changed): {np.mean(sparsity_scores)*100:.2f}%")
        print(f"Average Concept Change: {np.mean(concept_changes):.4f}")

    return {
        'success_rate': success_count/num_samples,
        'avg_l2_dist': total_l2_dist/success_count if success_count > 0 else float('inf'),
        'avg_l1_dist': total_l1_dist/success_count if success_count > 0 else float('inf'),
        'avg_sparsity': np.mean(sparsity_scores) if sparsity_scores else float('inf'),
        'avg_concept_change': np.mean(concept_changes) if concept_changes else float('inf')
    }


def generate_comprehensive_counterfactuals(model, test_loader, num_samples_per_class=1):
    """
    Generate and analyze counterfactuals for samples from each class
    the output format
    """
    device = next(model.parameters()).device
    all_results = []
    class_samples = {i: 0 for i in range(10)}  # For MNIST (0-9)
    
    model.eval()
    for images, labels in test_loader:
        # Check if we have enough samples for each class
        if all(count >= num_samples_per_class for count in class_samples.values()):
            break
            
        for img, label in zip(images, labels):
            label = label.item()
            if class_samples[label] >= num_samples_per_class:
                continue
                
            # Process single image
            img = img.unsqueeze(0).to(device).float()  # Ensure float tensor
            img.requires_grad_(True)  # Enable gradients
            
            # Generate counterfactuals for all other classes
            with torch.set_grad_enabled(True):  # Explicitly enable gradients
                orig_class, counterfactuals = model.get_counterfactual_explanations(
                    img, 
                    orig_class=label,
                    steps=100,#------------------------------------change to 100 for normal run
                    lr=0.01,
                    reg_lambda=0.1,
                    concept_lambda=1.0
                )
            
            # Store results
            result = {
                'original_image': img.detach().cpu(),
                'original_class': orig_class,
                'counterfactuals': [(tc, (cf.detach(), metrics)) for tc, (cf, metrics) in counterfactuals]
            }
            all_results.append(result)
            class_samples[label] += 1
                
    return all_results

def visualize_comprehensive_results(results):
    """
    Visualize counterfactuals in a 10x30 grid layout with horizontal bar charts
    first row: original image of source class and counterfactual images of all other classes
    second row: horizontal bar charts of concepts of source class and all other classes
    third row: difference map of source class and all other classes

    Difference Maps (Heat Maps):
        - Brighter/hotter colors (yellow/red) indicate areas where more changes were made
        - Darker colors (black/blue) indicate areas with minimal or no changes
        - This helps visualize which parts of the digit needed to be modified to change its classification

    """
    # Sort results by original class number
    results = sorted(results, key=lambda x: x['original_class'])
    
    for result in results:
        orig_img = result['original_image']
        orig_class = result['original_class']
        counterfactuals = result['counterfactuals']
        
        print(f"\n{'='*100}")
        print(f"Original Class: {orig_class}")
        print(f"{'='*100}")
        
        # Create a large figure for all classes
        fig = plt.figure(figsize=(25, 15))  # Wider figure to accommodate all classes
        plt.suptitle(f'Original Class {orig_class} vs All Other Classes', fontsize=16)
        
        # First, plot original image and its concepts
        # Original image
        plt.subplot(3, 10, 1)  # First column of first row
        plt.imshow(orig_img[0, 0].cpu().numpy(), cmap='gray')
        plt.title(f'Original\nClass {orig_class}')
        plt.axis('off')
        
        # Original concepts (horizontal bar chart)
        plt.subplot(3, 10, 11)  # First column of second row
        with torch.no_grad():
            _, (orig_concepts, orig_relevances), _ = model(orig_img.to(device))
        orig_concepts_data = orig_concepts[0,:,0].cpu().numpy()
        orig_relevances_data = orig_relevances[0,:,orig_class].cpu().numpy()
        concept_scores = orig_concepts_data * orig_relevances_data
        plt.barh(range(len(concept_scores)), concept_scores)
        plt.title(f'Original\nConcepts')
        
        # Empty plot for consistency
        plt.subplot(3, 10, 21)  # First column of third row
        plt.axis('off')
        plt.title('Original')
        
        # Create a dictionary of counterfactuals indexed by target class
        cf_dict = {tc: (cf, metrics) for tc, (cf, metrics) in counterfactuals}
        
        # Store metrics for compact display
        metrics_text = []
        
        # Plot each counterfactual in order
        col_idx = 1  # Start from second column
        for target_class in range(10):  # 0 through 9
            if target_class == orig_class:
                continue
                
            if target_class in cf_dict:
                cf, metrics = cf_dict[target_class]
                
                # Counterfactual image
                plt.subplot(3, 10, col_idx + 1)
                plt.imshow(cf[0, 0].cpu().numpy(), cmap='gray')
                plt.title(f'Class {target_class}')
                plt.axis('off')
                
                # Concept relevances (horizontal bar chart)
                plt.subplot(3, 10, col_idx + 11)
                concepts_cf = metrics['concepts_cf'][0,:,0].cpu().numpy()
                relevances_cf = metrics['relevances_cf'][0,:,target_class].cpu().numpy()
                concept_scores = concepts_cf * relevances_cf
                plt.barh(range(len(concept_scores)), concept_scores)
                plt.title(f'Concepts')
                
                # Difference map
                plt.subplot(3, 10, col_idx + 21)
                diff = torch.abs(cf - orig_img.to(device))[0, 0].cpu().numpy()
                plt.imshow(diff, cmap='hot')
                plt.title(f'Changes')
                plt.axis('off')
                
                # Collect metrics for compact display
                l2_dist = torch.norm(cf - orig_img.to(device)).item()
                sparsity = (torch.abs(cf - orig_img.to(device)) > 0.1).float().mean().item()
                metrics_text.append(
                    f"Class {orig_class}→{target_class}: L2={l2_dist:.2f}; Sparsity={sparsity*100:.1f}%; Concept Δ={metrics['concept_loss']:.2f}"
                )
            
            col_idx += 1
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for suptitle
        plt.show()
        
        # Print compact metrics in order
        print("\nMetrics Summary:")
        for metric in sorted(metrics_text, key=lambda x: int(x.split('→')[1].split(':')[0])):
            print(metric)
        print(f"{'='*100}\n")

"""
SENN Training Methods Documentation

1. Standard SENN Training (train_senn)
-----------------------------------
Purpose:
    Trains a Self-Explaining Neural Network focusing on robust concept learning 
    and classification accuracy.

Key Features:
    - Uses task loss and robustness loss
    - Optimizes for interpretable concept learning
    - More computationally efficient
    - Better for general classification tasks

Loss Components:
    - Task Loss: Classification accuracy
    - Robustness Loss: Ensures consistent explanations
    Total Loss = task_loss + rob_lambda * rob_loss

Use When:
    - Primary focus is classification accuracy
    - Need faster training times
    - Want robust concept learning
    - General interpretability is the goal
"""
num_epochs = 5  # Reduced for faster execution ---------------------------------------------------EPOCH CHANGE to 5 for normal run
batch_size = 64
num_concepts = 5
learning_rate = 0.001

# Load MNIST data
train_loader, test_loader = load_mnist_data(batch_size)

# Create model components
image_size = 28
image_channels = 1
num_classes = 10

# Instantiate the components
conceptizer = ConvConceptizer(
    image_size=image_size,
    num_concepts=num_concepts,
    concept_dim=1,
    image_channels=image_channels
)

parameterizer = ConvParameterizer(
    num_concepts=num_concepts,
    num_classes=num_classes,
    cl_sizes=(image_channels, 10, 20)
)

aggregator = SumAggregator(num_classes=num_classes)

# Create SENN model
model = SENN_CF(conceptizer, parameterizer, aggregator).to(device)

# Check if model exists and load, otherwise train
model_path = 'senn_mnist.pt'
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print("Training new model")
    # Configuration for both training methods
    # Initialize optimizer
    train_senn(model, train_loader, num_epochs)
    # torch.save(model.state_dict(), model_path)

# Evaluate model
accuracy = evaluate_senn(model, test_loader)

# Generate and evaluate counterfactuals
cf_metrics = generate_evaluate_counterfactuals(model, test_loader, num_samples=20)

# Example of generating a single counterfactual
# Get a test image
test_iter = iter(test_loader)
images, labels = next(test_iter)
test_image = images[0:1].to(device)
test_label = labels[0].item()

# Generate counterfactual for next class
target_class = (test_label + 1) % 10
_, counterfactuals = model.get_counterfactual_explanations(test_image, orig_class=test_label)

# Find the counterfactual for our target class
for cf_class, (counterfactual, metrics) in counterfactuals:
    if cf_class == target_class:
        success = True
        break
else:
    success = False
    counterfactual = None

if success:
    # Create visualization for single counterfactual
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(test_image[0, 0].cpu().numpy(), cmap='gray')
    plt.title(f'Original (Class {test_label})')
    plt.axis('off')
    
    # Counterfactual
    plt.subplot(1, 3, 2)
    plt.imshow(counterfactual[0, 0].cpu().numpy(), cmap='gray')
    plt.title(f'Counterfactual (Class {target_class})')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 3, 3)
    diff = torch.abs(counterfactual - test_image)[0, 0].cpu().numpy()
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.title('Changes Made')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    """
    - L2 Distance: Measures how different the counterfactual is from the original image
    - Sparsity: Percentage of pixels that were significantly changed
    - Concept Change: How much the underlying concepts changed in the transformation

"""
    print(f"\nCounterfactual Generation Metrics:")
    print(f"L2 Distance: {torch.norm(counterfactual - test_image).item():.4f}")
    print(f"Sparsity: {(torch.abs(counterfactual - test_image) > 0.1).float().mean().item()*100:.2f}%")
    print(f"Concept Change: {metrics['concept_loss']:.4f}")
else:
    print("Failed to generate counterfactual")

# Generate comprehensive counterfactuals
print("Generating comprehensive counterfactuals...")
results = generate_comprehensive_counterfactuals(model, test_loader, num_samples_per_class=1)

# Visualize results
print("\nVisualizing results for each class...")
visualize_comprehensive_results(results)

#everything  works
