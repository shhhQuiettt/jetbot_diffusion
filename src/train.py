import torch
import os
from torch import nn
from torch.nn.functional import mse_loss
from Unet1D import ConditionalUnet1D
from visual_encoder import MergingResnet18
from dataset import CarDataset
from dataclasses import dataclass
import logging
import random


@dataclass
class Config:
    sequence_features: int = 2
    visual_embedding_dim: int = 128
    time_embedding_dim: int = 128
    unet_down_dims: tuple = (256, 512, 1024)

    # past_horizont: int = 4
    past_horizont: int = 2
    # future_horizont: int = 6
    future_horizont: int = 2

    num_groups: int = 16
    
    no_guidance_probability: float = 0.1  
    guidance_scale: float = 1.0

    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 11
    patience: int = 10
    seed: int = 0xc0ffee

    img_channels: int = 3

    checkpoint_path: str = "./models"


class Trainer:
    def __init__(self, config: Config, guidence_embedding_model: nn.Module):
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.logger = logging.getLogger("Trainer")

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        self.guidance_embedding_model = guidence_embedding_model.to(self.device)
        # self.null_visual_embedding = torch.randn(config.visual_embedding_dim, device=self.device)  
        self.null_visual_embedding = nn.Parameter(torch.randn(config.visual_embedding_dim, device=self.device), requires_grad=True)

        self.unet = ConditionalUnet1D(
            sequence_features_dim=config.sequence_features,
            time_dim=config.time_embedding_dim,
            guidance_dim=config.visual_embedding_dim,
            down_dims=config.unet_down_dims
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.guidance_embedding_model.parameters()) + list(self.unet.parameters()),
            lr=config.learning_rate
        )

        self.train_losses = []

        if not os.path.exists(config.checkpoint_path):
            os.makedirs(config.checkpoint_path)


    def alpha(self, t):
        return t.unsqueeze(-1).unsqueeze(-1)

    def alpha_dot(self, t):
        return torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def beta(self, t):
        return (1.0-t).unsqueeze(-1).unsqueeze(-1)

    def beta_dot(self, t):
        return -torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def train(self, train_loader, val_loader):

        for epoch in range(self.config.num_epochs):

            self.guidance_embedding_model.train()
            self.unet.train()

            for batch in train_loader:
                images, sequences = batch
                images, sequences = images.to(self.device), sequences.to(self.device)

                assert sequences.shape[1] == self.config.future_horizont, f"Expected sequence length {self.config.future_horizont}, got {sequences.shape[1]}"
                assert sequences.shape[2] == self.config.sequence_features, f"Expected sequence features {self.config.sequence_features}, got {sequences.shape[2]}"

                self.optimizer.zero_grad()

                noise = torch.randn_like(sequences, device=self.device)  
                times = torch.rand(sequences.size(0), device=self.device)

                x_t = self.alpha(times) * sequences + self.beta(times) * noise

                no_visual_embedding_mask = torch.rand(x_t.size(0), device=self.device) < self.config.no_guidance_probability

                visual_embeddings = self.guidance_embedding_model(images)
                visual_embeddings[no_visual_embedding_mask] = self.null_visual_embedding

                predicted_flow = self.unet(x_t, times, visual_embeddings)

                loss = mse_loss(predicted_flow, (self.alpha_dot(times) * sequences + self.beta_dot(times) * noise))

                self.logger.info("=" * 50)
                self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Batch Loss: {loss.item():.4f}")
                self.logger.info(f"Train loss delta: {loss.item() - self.train_losses[-1] if len(self.train_losses) > 0 else 0.0:.4f}")
                self.logger.info("=" * 50)

                self.train_losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

            if (epoch+1) % 5 == 0:
                checkpoint_file = os.path.join(self.config.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
                self.save_model(checkpoint_file)
                self.logger.info(f"Saved checkpoint: {checkpoint_file}")

    def save_model(self, path):
        torch.save({
            'guidance_embedding_model_state_dict': self.guidance_embedding_model.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'config': self.config,
            'null_visual_embedding': self.null_visual_embedding.detach().cpu(),
        }, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config()
    guidance_model = MergingResnet18(embedding_size=config.visual_embedding_dim, input_channels=config.img_channels, num_images=config.past_horizont, num_groups=config.num_groups).to("cpu")
    trainer = Trainer(config, guidance_model)

    dataset = CarDataset("./dataset/", device="cpu", past_horizont=config.past_horizont, future_horizont=config.future_horizont)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    trainer.train(train_loader=dataloader, val_loader=None)

