#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    [2] https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vq_vae.py#L2
    """
    def __init__(self, config, device):
        super(VectorQuantizer, self).__init__()
        
        self.config = config
        self.device = device
        
        self.embedding = nn.Embedding(config["num_embeddings"], config["embedding_dim"]).to(device)
        self.embedding.weight.data.uniform_(-1/config["num_embeddings"], 1/config["num_embeddings"]).to(device)
        
    def forward(self, latents):
        latents = latents.to(self.device)
        latents = latents.permute(0, 2, 3, 1).contiguous() # [B x D x H x W] -> [B x H x W x D]
        flat_latents = latents.view(-1, self.config["embedding_dim"]) # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())

        # Get the encoding that has the min distance
        encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_index.size(0), self.config["num_embeddings"]).to(self.device)
        # dim: one hot encoding dimension
        # index: position of src
        # value: the value of corresponding index
        encoding_one_hot.scatter_(dim=1, index=encoding_index, value=1)

        # Quantize the latents
        quantized_latent = torch.matmul(encoding_one_hot, self.embedding.weight) # [BHW, D]
        quantized_latent = quantized_latent.view(latents.shape)

        # Compute the VQ losses
        embedding_loss = F.mse_loss(latents.detach(), quantized_latent) # training embeddings
        commitment_loss = F.mse_loss(latents, quantized_latent.detach()) # prevent encoder from growing
        vq_loss = embedding_loss + self.config["beta"] * commitment_loss

        # Add the residue back to the latents (straight-through gradient estimation)
        quantized_latent = latents + (quantized_latent - latents).detach()
        
        return quantized_latent.permute(0, 3, 1, 2).contiguous(), vq_loss # [B x D x H x W]
#%%
class ResidualLayer(nn.Module):
    """
    Reference:
    [1] https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vq_vae.py#L2
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, out_channels,
                                            kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)
#%%
class VQVAE(nn.Module):
    """
    Reference:
    [1] https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vq_vae.py#L2
    """
    def __init__(self, config, device, in_channels=3, hidden_dims=[128, 256]):
        super(VQVAE, self).__init__()

        self.config = config
        self.device = device
        
        modules = []

        """Build Encoder"""
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, self.config["embedding_dim"],
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules).to(device)

        """Build Vector Quantizer"""
        self.vq_layer = VectorQuantizer(self.config, device).to(device)

        """Build Decoder"""
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.config["embedding_dim"],
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        # hidden_dims.reverse()
        hidden_dims_reversed = list(reversed(hidden_dims))

        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i],
                                       hidden_dims_reversed[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules).to(device)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input):
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]
    
    # def loss_function(self, *args):
    #     """
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     recons = args[0]
    #     input = args[1]
    #     vq_loss = args[2]

    #     recons_loss = F.mse_loss(recons, input)

    #     loss = recons_loss + vq_loss
    #     return {'loss': loss,
    #             'Reconstruction_Loss': recons_loss,
    #             'VQ_Loss':vq_loss}

    """FIXME"""
    # def sample(self,
    #            num_samples: int,
    #            current_device: Union[int, str], **kwargs) -> Tensor:
    #     raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
#%%
def main():
    config = {
        "n": 10,
        "num_embeddings": 256,
        "embedding_dim": 32,
        "beta": 0.25,
    }
    
    model = VQVAE(config, 'cpu')
    for x in model.parameters():
        print(x.shape)
        
    batch = torch.rand(config["n"], 3, 32, 32)
    xhat, _, vq_loss = model(batch)
    encoding = model.encode(batch)
    quantized_inputs, vq_loss = model.vq_layer(encoding)
    
    assert encoding.shape == (config["n"], config["embedding_dim"], 8, 8)
    assert quantized_inputs.shape == (config["n"], config["embedding_dim"], 8, 8)
    assert xhat.shape == (config["n"], 3, 32, 32)
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
#%%
# config = {
#     "num_embeddings": 256,
#     "embedding_dim": 32,
#     "beta": 0.25,
# }

# embedding = nn.Embedding(config["num_embeddings"], config["embedding_dim"])
# embedding.weight.data.uniform_(-1/config["num_embeddings"], 1/config["num_embeddings"])

# latents = torch.randn(10, config["embedding_dim"], 4, 4)
# latents = latents.permute(0, 2, 3, 1).contiguous() # [B x D x H x W] -> [B x H x W x D]
# flat_latents = latents.view(-1, config["embedding_dim"]) # [BHW x D]

# # Compute L2 distance between latents and embedding weights
# dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#     torch.sum(embedding.weight ** 2, dim=1) - \
#     2 * torch.matmul(flat_latents, embedding.weight.t())

# # Get the encoding that has the min distance
# encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)

# # Convert to one-hot encodings
# encoding_one_hot = torch.zeros(encoding_index.size(0), config["num_embeddings"])
# # dim: one hot encoding dimension
# # index: position of src
# # value: the value of corresponding index
# encoding_one_hot.scatter_(dim=1, index=encoding_index, value=1)

# # Quantize the latents
# quantized_latent = torch.matmul(encoding_one_hot, embedding.weight) # [BHW, D]
# quantized_latent = quantized_latent.view(latents.shape)

# # Compute the VQ losses
# embedding_loss = F.mse_loss(latents.detach(), quantized_latent) # training embeddings
# commitment_loss = F.mse_loss(latents, quantized_latent.detach()) # prevent encoder from growing
# vq_loss = embedding_loss + config["beta"] * commitment_loss

# # Add the residue back to the latents (straight-through gradient estimation)
# quantized_latent = latents + (quantized_latent - latents).detach()

# quantized_latent = quantized_latent.permute(0, 3, 1, 2).contiguous()