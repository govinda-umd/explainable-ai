import torch 
import torch.nn as nn


class GRU_classifier(nn.Module):
    def __init__(self, args):
        super(GRU_classifier, self).__init__()

        self.MASK = args.MASK

        self.gru = nn.GRU(
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            batch_first=True,
            # dropout=args.dropout
        )
        
        self.fc = nn.Linear(args.hidden_dim, args.out_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x, hn = self.gru(x)
        x = self.dropout(x)
        y = self.fc(x)
        return y

    # def initHidden(self, args):
        # return torch.zeros(1, args.batch_size, args.num_units, device=device)

    def accuracy(self, y_pred, y_true):
        labels_pred = torch.argmax(y_pred, axis=-1)
        mask = (y_true != self.MASK).bool()
        correct = 1-torch.abs(labels_pred[mask] - y_true[mask])
        return correct.sum() / len(correct)

class Softmax_Model(nn.Module):
    def __init__(self, base_model, temperature):
        super(Softmax_Model, self).__init__()

        self.base_model = base_model
        self.temperature = temperature 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.base_model(x)
        y = self.softmax(y / self.temperature)
        return y #.view(-1, y.shape[-1])

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.MASK = args.MASK

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(args.hidden_dim, args.out_dim),
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def accuracy(self, y_pred, y_true):
        labels_pred = torch.argmax(y_pred, axis=-1)
        mask = (y_true != self.MASK).bool()
        correct = 1-torch.abs(labels_pred[mask] - y_true[mask])
        return correct.sum() / len(correct)
    