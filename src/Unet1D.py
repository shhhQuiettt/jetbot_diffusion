import torch
import torch.nn as nn
import math

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: b
        Returns:
        - embeddings: b d
        """
        print(f"{t.shape=}")
        print(f"{self.weights.shape=}")
        frequencies = 2 * torch.pi * self.weights * t.unsqueeze(1)

        sin_part = torch.sin(frequencies)
        cos_part = torch.cos(frequencies)
        assert sin_part.shape[0] == cos_part.shape[0], f"Expected sin_part and cos_part to have the same batch size, got {sin_part.shape[0]} and {cos_part.shape[0]}"

        embedding = torch.cat([sin_part, cos_part], dim=-1)
        embedding = embedding.view(t.shape[0], self.dim)

        assert embedding.shape == (t.shape[0], self.dim ), f"Expected embedding shape to be {t.shape[0], self.dim}, got {embedding.shape}"

        return embedding

class SinusoidalPosEmb(nn.Module):
    """1D Sinusoidal Positional Embedding for Diffusion Timesteps"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        print(f"{emb.shape=}")
        return emb

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.Mish()
        
        # FiLM projection: outputs scale and shift parameters
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.Mish()
        
        # Skip connection projection if dimensions change
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # First Conv layer
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        # Feature-wise Linear Modulation (FiLM)
        # cond shape: [B, cond_dim] -> [B, 2 * out_channels] -> [B, 2 * out_channels, 1]
        fiLM_params = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = fiLM_params.chunk(2, dim=1)
        out = out * (scale + 1.0) + shift # Modulation
        
        # Second Conv layer
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        return out + self.residual_proj(x)

class ConditionalUnet1D(nn.Module):
    """Main 1D U-Net Architecture for Action Diffusion (PATCHED)"""
    def __init__(self, action_dim, obs_dim, embed_dim=256, down_dims=[256, 512, 1024]):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # 1. Timestep Embedding setup
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(embed_dim),
            FourierEncoder(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        cond_dim = embed_dim + obs_dim
        
        # Initial convolution mapping action_dim to first hidden dim
        self.init_conv = nn.Conv1d(action_dim, down_dims[0], kernel_size=3, padding=1)
        
        # 2. Encoder (Downsampling)
        self.down_blocks = nn.ModuleList()
        in_out = list(zip(down_dims[:-1], down_dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim),
                nn.Conv1d(dim_out, dim_out, kernel_size=4, stride=2, padding=1) if not is_last else nn.Identity()
            ]))
            
        # 3. Bottleneck
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim)
        
        # 4. Decoder (Upsampling)
        self.up_blocks = nn.ModuleList()
        in_out = list(zip(down_dims[::-1][:-1], down_dims[::-1][1:]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.up_blocks.append(nn.ModuleList([
                # FIX: Map from dim_in * 2 (due to skip connection) directly down to dim_out
                ConditionalResidualBlock1D(dim_in * 2, dim_out, cond_dim), 
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim),
                # FIX: Upsampling operates on dim_out channels now
                nn.ConvTranspose1d(dim_out, dim_out, kernel_size=4, stride=2, padding=1) if not is_last else nn.Identity()
            ]))
            
        # Final projection back to action dimensions
        self.final_conv = nn.Sequential(
            ConditionalResidualBlock1D(down_dims[0] * 2, down_dims[0], cond_dim),
            nn.Conv1d(down_dims[0], action_dim, kernel_size=1)
        )

    def forward(self, x, timestep, global_cond):
        t_emb = self.time_mlp(timestep)
        cond = torch.cat([t_emb, global_cond], dim=-1)
        
        # Pass through initial conv
        x = self.init_conv(x)
        
        # FIX: We must save the output of the init_conv to match the final decoder layer!
        hiddens = [x] 
        
        # Encoder
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond)
            x = res2(x, cond)
            hiddens.append(x)
            x = downsample(x)
            
        # Bottleneck
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)
        
        # Decoder
        for res1, res2, upsample in self.up_blocks:
            h = hiddens.pop()
            x = torch.cat([x, h], dim=1) 
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)
            
        # Final layers
        h = hiddens.pop() # This will no longer be empty!
        x = torch.cat([x, h], dim=1)
        x = self.final_conv[0](x, cond)
        x = self.final_conv[1](x)
        
        return x



def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    return total_params

def get_model_flops(model, dummy_input, timestep, global_cond):
    from thop import profile
    flops, params = profile(model, inputs=(dummy_input, timestep, global_cond), verbose=False)
    print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    return flops

def get_model_storage_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # Assuming 32-bit floats
    print(f"Total storage size: {total_size_bytes / 1e6:.2f} MB")
    return total_size_bytes


if __name__ == "__main__":
    batch_size = 4
    action_dim = 2
    obs_dim = 128
    seq_len = 32
    model = ConditionalUnet1D(action_dim, obs_dim, down_dims=[256, 512, 1024])
    x = torch.randn(batch_size, action_dim, seq_len)
    timestep = torch.randint(0, 1000, (batch_size,))
    global_cond = torch.randn(batch_size, obs_dim)
    output = model(x, timestep, global_cond)

    print("Output shape:", output.shape)  # Should be [batch_size, action_dim, seq_len]
    # print(output)
    # print("Model summary:")

    get_model_params(model)
    get_model_flops(model, x, timestep, global_cond)
    get_model_storage_size(model)



