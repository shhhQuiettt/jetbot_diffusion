from torch import nn
import torch
import torchsummary

class DiffusionModel(nn.Module):
    def __init__(self,command_sequence_length: int, time_embedding_dim: int, visual_embedding_dim: int):
        super().__init__()

        self.command_sequence_length = command_sequence_length
        self.time_embedding_dim = time_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim

        # input dimention is (batch_size, 2, command_sequence_length) 
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.leaky_relu2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * (command_sequence_length - 4) + time_embedding_dim + visual_embedding_dim, 128)
        self.leaky_relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.leaky_relu4 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, command_sequence_length * 2)


    def forward(self, x, time_embedding, visual_embedding):
        assert time_embedding.shape[1] == self.time_embedding_dim, f"Expected time_embedding dimension {self.time_embedding_dim}, got {time_embedding.shape[1]}"
        assert visual_embedding.shape[1] == self.visual_embedding_dim, f"Expected visual_embedding dimension {self.visual_embedding_dim}, got {visual_embedding.shape[1]}"
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.flatten(x)
        x = torch.cat((x, time_embedding, visual_embedding), dim=1)
        x = self.fc1(x)
        x = self.leaky_relu3(x)
        x = self.fc2(x)
        x = self.leaky_relu4(x)
        x = self.fc3(x)
        x = x.view(-1, 2, self.command_sequence_length)
        return x



if __name__ == "__main__": 
    time_embedding_dim = 16
    visual_embedding_dim = 32
    command_sequence_length = 10
    model = DiffusionModel(command_sequence_length, time_embedding_dim, visual_embedding_dim)

    # Example input
    batch_size = 4
    input_data = torch.randn(batch_size, 2, command_sequence_length)
    time_embedding = torch.randn(batch_size, time_embedding_dim)
    visual_embedding = torch.randn(batch_size, visual_embedding_dim)

    # torchsummary.summary(model, [(2, command_sequence_length), (time_embedding_dim,), (visual_embedding_dim,)], device='cpu')

    output = model(input_data, time_embedding, visual_embedding)
    print("Output shape:", output.shape)  # Expected shape: (batch_size,
    # 2, command_sequence_length)



