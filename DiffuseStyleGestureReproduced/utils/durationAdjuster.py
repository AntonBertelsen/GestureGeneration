from torch import nn

class DurationAdjuster(nn.Module):
    def __init__(self, input_length, target_length):
        super().__init__()

        # Define a net to adjust the sequence length
        self.net = nn.Sequential(
            nn.Linear(input_length, target_length),
            nn.ReLU(inplace=True),
            nn.Linear(target_length, target_length),
            nn.ReLU(inplace=True),
            nn.Linear(target_length, target_length)
        )

    def forward(self, x, target_length):
        batch_size, seq_length, feature_dim = x.shape
        
        # if seq_length == target_length:
            # return x        
        
        # Reshape for the linear layer operation
        # [batch_size, seq_length, feature_dim] -> [batch_size, feature_dim, seq_length]
        # x_transposed = x.transpose(1, 2)
        
        # Apply linear transformation to adjust sequence length
        # [batch_size, feature_dim, seq_length] -> [batch_size, feature_dim, target_length]
        # adjusted = self.net(x_transposed)
        
        # Reshape back to original format
        # [batch_size, feature_dim, target_length] -> [batch_size, target_length, feature_dim]
        # return adjusted.transpose(1, 2)

        # just crop the sequence length (from the end)
        return x[:, :target_length, :]