import torch
import torch.nn as nn

class BehaviorNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(BehaviorNetModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(128, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  
        x = x.float()
        rnn_out, _ = self.gru(x)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        weighted_rnn_out = torch.sum(attention_weights * rnn_out, dim=1)
        x = self.relu(self.fc1(weighted_rnn_out))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
