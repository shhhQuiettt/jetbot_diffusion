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
    past_horizont: int = 8
    # future_horizont: int = 6
    future_horizont: int = 32

    num_groups: int = 16

    no_guidance_probability: float = 0.1
    guidance_scale: float = 1.0

    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 11
    patience: int = 10
    seed: int = 0xC0FFEE

    img_channels: int = 3

    checkpoint_path: str = "./models"


class Trainer:
    def __init__(
        self,
        config: Config,
        guidence_embedding_model: nn.Module,
        *,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ):
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.logger = logging.getLogger("Trainer")

        self.config = config
        self.dataset_train = dataloader_train
        self.dataset_val = dataloader_val

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        self.guidance_embedding_model = guidence_embedding_model.to(self.device)
        # self.null_visual_embedding = torch.randn(config.visual_embedding_dim, device=self.device)
        self.null_visual_embedding = nn.Parameter(
            torch.randn(config.visual_embedding_dim, device=self.device),
            requires_grad=True,
        )

        self.unet = ConditionalUnet1D(
            sequence_features_dim=config.sequence_features,
            time_dim=config.time_embedding_dim,
            guidance_dim=config.visual_embedding_dim,
            down_dims=config.unet_down_dims,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.guidance_embedding_model.parameters())
            + list(self.unet.parameters()),
            lr=config.learning_rate,
        )

        self.train_losses = []
        self.val_losses = []

        if not os.path.exists(config.checkpoint_path):
            os.makedirs(config.checkpoint_path)

    def alpha(self, t):
        return t.unsqueeze(-1).unsqueeze(-1)

    def alpha_dot(self, t):
        return torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def beta(self, t):
        return (1.0 - t).unsqueeze(-1).unsqueeze(-1)

    def beta_dot(self, t):
        return -torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def train(self):
        last_val_loss = float("inf")
        for epoch in range(self.config.num_epochs):
            self.guidance_embedding_model.train()
            self.unet.train()

            running_loss = 0.0
            num_batches = 0

            for batch in self.dataset_train:
                images, sequences = batch
                images, sequences = images.to(self.device), sequences.to(self.device)

                assert sequences.shape[2] == self.config.future_horizont, (
                    f"Expected sequence length {self.config.future_horizont}, got {sequences.shape[1]}"
                )
                assert sequences.shape[1] == self.config.sequence_features, (
                    f"Expected sequence features {self.config.sequence_features}, got {sequences.shape[2]}"
                )

                self.optimizer.zero_grad()

                noise = torch.randn_like(sequences, device=self.device)
                times = torch.rand(sequences.size(0), device=self.device)

                x_t = self.alpha(times) * sequences + self.beta(times) * noise

                no_visual_embedding_mask = (
                    torch.rand(x_t.size(0), device=self.device)
                    < self.config.no_guidance_probability
                )

                visual_embeddings = self.guidance_embedding_model(images)
                visual_embeddings[no_visual_embedding_mask] = self.null_visual_embedding

                predicted_flow = self.unet(x_t, times, visual_embeddings)

                loss = mse_loss(
                    predicted_flow,
                    (self.alpha_dot(times) * sequences + self.beta_dot(times) * noise),
                )

                running_loss += loss.item()
                num_batches += 1

                loss.backward()
                self.optimizer.step()

            assert num_batches > 0, "No batches were processed during training"
            average_loss = running_loss / num_batches
            self.train_losses.append(average_loss)

            val_loss = self.validate()

            val_loss_delta = val_loss - last_val_loss
            last_val_loss = val_loss
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.num_epochs}] - Train Loss: {self.train_losses[-1]:.4f} - Val Loss: {val_loss:.4f} - Val Loss Delta: {val_loss_delta:.4f}"
            )

            if (epoch + 1) % 5 == 0:
                checkpoint_file = os.path.join(
                    self.config.checkpoint_path, f"checkpoint_epoch_{epoch + 1}.pth"
                )
                self.save_model(checkpoint_file)
                self.logger.info(f"Saved checkpoint: {checkpoint_file}")

    def save_model(self, path):
        torch.save(
            {
                "guidance_embedding_model_state_dict": self.guidance_embedding_model.state_dict(),
                "unet_state_dict": self.unet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "config": self.config,
                "null_visual_embedding": self.null_visual_embedding.detach().cpu(),
            },
            path,
        )

    def validate(self):
        self.guidance_embedding_model.eval()
        self.unet.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataset_val:
                images, sequences = batch
                images, sequences = images.to(self.device), sequences.to(self.device)

                noise = torch.randn_like(sequences, device=self.device)
                times = torch.rand(sequences.size(0), device=self.device)

                x_t = self.alpha(times) * sequences + self.beta(times) * noise

                visual_embeddings = self.guidance_embedding_model(images)

                predicted_flow = self.unet(x_t, times, visual_embeddings)

                loss = mse_loss(
                    predicted_flow,
                    (self.alpha_dot(times) * sequences + self.beta_dot(times) * noise),
                )

                total_loss += loss.item()
                num_batches += 1

        assert num_batches > 0, "No batches were processed during validation"
        average_loss = total_loss / num_batches

        return average_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config()
    guidance_model = MergingResnet18(
        embedding_size=config.visual_embedding_dim,
        input_channels=config.img_channels,
        num_images=config.past_horizont,
        num_groups=config.num_groups,
    ).to("cpu")

    dataset = CarDataset(
        "./dataset/",
        device="cpu",
        past_horizont=config.past_horizont,
        future_horizont=config.future_horizont,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=False
    )

    trainer = Trainer(
        config,
        guidance_model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )

    trainer.train()
