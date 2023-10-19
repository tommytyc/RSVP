from transformers import AutoModel
import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_text):
        hidden_states = self.transformer(**input_text)[1]
        # hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits, hidden_states
