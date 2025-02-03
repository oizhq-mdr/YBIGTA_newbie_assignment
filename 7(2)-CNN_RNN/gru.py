import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        
        # Update gate
        self.w_z = nn.Linear(input_size, hidden_size)
        self.u_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset gate
        self.w_r = nn.Linear(input_size, hidden_size)
        self.u_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # New memory
        self.w_h = nn.Linear(input_size, hidden_size)
        self.u_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # Update gate
        z = torch.sigmoid(self.w_z(x) + self.u_z(h))
        
        # Reset gate
        r = torch.sigmoid(self.w_r(x) + self.u_r(h))
        
        # New memory
        h_tilde = torch.tanh(self.w_h(x) + self.u_h(r * h))
        
        # Output
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        # Process sequence
        for t in range(seq_length):
            x_t = inputs[:, t, :]  # Current input
            h = self.cell(x_t, h)  # Update hidden state
            
        # Return final hidden state
        return h