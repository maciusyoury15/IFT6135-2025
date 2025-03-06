from torch import nn
import matplotlib.pyplot as plt
import os
import numpy as np

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size =patch_size, stride =patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.mlp_tokens(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.mlp_channels(y)
        x = x + y
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        out = self.patchemb(images)
        # step 2 Go through the mixer blocks
        out = self.blocks(out)
        # step 3 go through layer norm
        out = self.norm(out)
        # step 4 Global averaging spatially
        out = out.mean(dim=1)
        # Classification
        logits = self.head(out)
        return logits
    
    def visualize(self, logdir: str, layer_idx: int = 0):
        """ Visualize the token mixer layer in the desired directory """
        # Ensure the directory exists
        os.makedirs(logdir, exist_ok=True)

        # Select the specified MixerBlock
        if layer_idx >= len(self.blocks):
            raise ValueError(f"Invalid layer index {layer_idx}. Must be in range 0 to {len(self.blocks) - 1}.")

        mixer_block = self.blocks[layer_idx]

        # Extract the weight matrix from the token mixing MLP
        token_mixer_weights = mixer_block.mlp_tokens.fc1.weight.data.cpu().numpy()

        # Normalize weights for visualization
        min_val, max_val = token_mixer_weights.min(), token_mixer_weights.max()
        token_mixer_weights = (token_mixer_weights - min_val) / (max_val - min_val)

        # Plot the token mixing matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(token_mixer_weights, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"Token Mixing Weights - Layer {layer_idx}")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")

        # Save the visualization
        save_path = os.path.join(logdir, f"token_mixer_layer_{layer_idx}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Visualization saved at {save_path}")

        #  Additional analysis
        print("Weight Matrix Shape:", token_mixer_weights.shape)
        print("Weight Statistics:")
        print("Mean:", np.mean(token_mixer_weights))
        print("Standard Deviation:", np.std(token_mixer_weights))
        print("Min:", np.min(token_mixer_weights))
        print("Max:", np.max(token_mixer_weights))
