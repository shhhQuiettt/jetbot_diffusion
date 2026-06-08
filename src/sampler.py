from torch.nn.functional import alpha_dropout
from Unet1D import ConditionalUnet1D
from visual_encoder import MergingResnet18
from dataclasses import dataclass
from pprint import pprint
import torch
from train import Config


@dataclass
class SamplerConfig:
    num_steps: int = 1024
    sigma: float = 0.0
    cfg_strength: float = 1.0


class Sampler:
    def __init__(self, conditional_vecto_field_model_path: str, config: SamplerConfig):
        self.num_steps = config.num_steps
        self.dt = 1.0 / self.num_steps
        self.sigma = config.sigma
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(conditional_vecto_field_model_path, weights_only=False)
        cvf_config: Config = state_dict["config"]
        pprint(f"Loaded model with config: {cvf_config}")

        self.cvf_config = cvf_config

        self.guidance_embedding_model = MergingResnet18(
            num_images=cvf_config.past_horizont,
            input_channels=cvf_config.img_channels,
            embedding_size=cvf_config.visual_embedding_dim,
            num_groups=cvf_config.visual_embedding_dim,
        )
        self.guidance_embedding_model.load_state_dict(
            state_dict["guidance_embedding_model_state_dict"],
        )
        self.guidance_embedding_model.eval()

        self.unet = ConditionalUnet1D(
            sequence_features_dim=cvf_config.past_horizont,
            time_dim=cvf_config.time_embedding_dim,
            guidance_dim=cvf_config.visual_embedding_dim,
            down_dims=cvf_config.unet_down_dims,
        )
        self.unet.load_state_dict(state_dict["unet_state_dict"])
        self.null_visual_embedding = state_dict["null_visual_embedding"].to(self.device)
        self.unet.eval()

    def sample(self, guidance_images: torch.Tensor) -> torch.Tensor:
        assert (
            guidance_images.shape[1]
            == self.guidance_embedding_model.num_images
            == self.cvf_config.past_horizont
        ), (
            f"Past horizont mismatch {guidance_images.shape[1]} vs {self.guidance_embedding_model.num_images} vs {self.cvf_config.past_horizont}"
        )

        batch_size = guidance_images.shape[0]

        with torch.no_grad():
            guidance_embedding = self.guidance_embedding_model(guidance_images)
            x = torch.randn(
                (
                    batch_size,
                    self.cvf_config.future_horizont,
                    self.cvf_config.sequence_features,
                ),
                device=guidance_images.device,
            )
            t = 0.0

            for i in range(self.num_steps):
                predicted_guided_vecor_field = self.unet(
                    x,
                    t * torch.ones(batch_size, device=guidance_images.device),
                    guidance_embedding,
                )
                predicted_null_vecotr_field = 0.0  # self.unet(x, t*torch.ones(batch_size, device=guidance_images.device), self.null_visual_embedding.expand(batch_size, -1))

                predicted_vector_field = (
                    (1 - self.config.cfg_strength) * predicted_null_vecotr_field
                    + self.config.cfg_strength * predicted_guided_vecor_field
                )

                x += (
                    (1 + self.sigma**2 / (2 * self.a(t))) * predicted_vector_field
                    - (self.sigma**2 * self.b(t)) / (2 * self.a(t)) * x
                ) * self.dt + self.sigma * torch.sqrt(self.dt) * torch.randn_like(x)

                t += self.dt

        return x

    def alpha(self, t):
        return t.unsqueeze(-1).unsqueeze(-1)

    def alpha_dot(self, t):
        return torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def beta(self, t):
        return (1.0 - t).unsqueeze(-1).unsqueeze(-1)

    def beta_dot(self, t):
        return -torch.ones_like(t, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def a(self, t):
        return self.beta(t) ** 2 * self.alpha_dot(t) / self.alpha(t) - self.beta(
            t
        ) * self.beta_dot(t)

    def b(self, t):
        return self.alpha_dot(t) / self.alpha(t)


if __name__ == "__main__":
    from dataset import CarDataset

    config = SamplerConfig()
    model_path = "models/checkpoint_epoch_10.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = Sampler(model_path, config)

    dataset = CarDataset(
        "./dataset/", device=device, future_horizont=2, past_horizont=2
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    batch = next(iter(dataloader))
    images, sequences = batch

    sampled_sequences = sampler.sample(images)
    print(sampled_sequences.shape)
