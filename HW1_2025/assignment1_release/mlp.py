import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
            :param input: Tensor of shape [bsz, in_features]
            :return result: Tensor of shape [bsz, out_features]
        """
        # Apply the linear transformation: y = xA^T + b
        return torch.nn.functional.linear(input, self.weight, self.bias)


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(self.input_size, self.hidden_sizes, self.num_classes)
        
        # Initialization
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        hidden_layers = nn.ModuleList()

        # Building hidden layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            hidden_layer = Linear(prev_size, hidden_size)
            hidden_layers.append(self.activation_fn(self.activation, hidden_layer))
            prev_size = hidden_size

        # Build output layer
        output_layer = nn.Linear(prev_size, num_classes)

        return hidden_layers, output_layer

    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        return activation(inputs)
        
    def _initialize_linear_layer(self, module: Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        if isinstance(module, Linear):  # Ensure it's a Linear layer
            nn.init.xavier_normal_(module.weight)  # Glorot Normal initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Set bias to zero
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        batch_size = images.shape[0]
        x = images.view(batch_size, -1) # Flatten images to [batch, in_features]

        for layer in self.hidden_layers:
            x = layer(x)   # Pass through layers

        logits = self.output_layer(x)   # Final output layer
        return logits
