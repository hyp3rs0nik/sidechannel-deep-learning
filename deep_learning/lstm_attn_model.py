import torch
import torch.nn as nn

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(AttentionLSTMModel, self).__init__()
        # LSTM for sequential modeling
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_size)
        rnn_out, _ = self.rnn(x)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        weighted_rnn_out = torch.sum(attention_weights * rnn_out, dim=1)
        x = self.relu(self.fc1(weighted_rnn_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
