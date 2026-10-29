import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        def conv_relu(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *conv_relu(1, 64),
            nn.MaxPool2d(2, 2),
            *conv_relu(64, 128),
            nn.MaxPool2d(2, 2),
            *conv_relu(128, 256),
            *conv_relu(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            *conv_relu(256, 512),
            *conv_relu(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        log_probs = logits.log_softmax(2)
        return log_probs
