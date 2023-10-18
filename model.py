from transformers import MT5ForConditionalGeneration, MT5EncoderModel, MT5Tokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
SUBCAT_DIM = 64

def sgpt_sentence_emb(batch_tokens, last_hidden_state):
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
    embeddings = sum_embeddings / sum_mask
    return embeddings

def last_tok_sentence_emb(batch_tokens, last_hidden_state):
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )
    last_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    return last_embeddings

class Classifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
        self.fuse_dense = nn.Linear(768*2+SUBCAT_DIM, 768)
        self.subcat_dense = nn.Linear(768, SUBCAT_DIM)
        
    def forward(self, input_text, response_text, subcat_text):
        # self.bert.gradient_checkpointing_enable()
        text_outputs = self.bert(**input_text)
        response_outputs = self.bert(**response_text)
        subcat_outputs = self.bert(**subcat_text)
        text_pooled_output = text_outputs[1]
        response_pooled_output = response_outputs[1]
        subcat_pooled_output = subcat_outputs[1]

        text_pooled_output = self.dropout(text_pooled_output)
        response_pooled_output = self.dropout(response_pooled_output)
        subcat_pooled_output = self.dropout(subcat_pooled_output)
        subcat_pooled_output = self.subcat_dense(subcat_pooled_output)
        pooled_output = self.fuse_dense(torch.cat([text_pooled_output, response_pooled_output, subcat_pooled_output], 1))
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return pooled_output, logits

class ClassifierLabelKLD(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ClassifierLabelKLD, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
        self.fuse_dense = nn.Linear(768*2+SUBCAT_DIM, 768)
        self.subcat_dense = nn.Linear(768, SUBCAT_DIM)
        self.label_dense = nn.Linear(768, 768)
        
    def forward(self, input_text, response_text, subcat_text, label_text=None):
        text_outputs = self.bert(**input_text)
        response_outputs = self.bert(**response_text)
        subcat_outputs = self.bert(**subcat_text)
        text_pooled_output = text_outputs[1]
        response_pooled_output = response_outputs[1]
        subcat_pooled_output = subcat_outputs[1]
        
        text_pooled_output = self.dropout(text_pooled_output)
        response_pooled_output = self.dropout(response_pooled_output)
        subcat_pooled_output = self.dropout(subcat_pooled_output)
        subcat_pooled_output = self.subcat_dense(subcat_pooled_output)
        pooled_output = self.fuse_dense(torch.cat([text_pooled_output, response_pooled_output, subcat_pooled_output], 1))
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        label_pooled_output, label_logits = None, None
        if label_text is not None:
            with torch.no_grad():
                label_outputs = self.bert(**label_text)
                label_pooled_output = label_outputs[1]
                label_pooled_output = self.dropout(label_pooled_output)
                label_pooled_output = self.label_dense(label_pooled_output)
                label_pooled_output = torch.tanh(label_pooled_output)
                label_pooled_output = self.dropout(label_pooled_output)
            
        return pooled_output, label_pooled_output, logits

class RoleClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(RoleClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
        self.fuse_dense = nn.Linear(768*2+SUBCAT_DIM, 768)
        self.subcat_dense = nn.Linear(768, SUBCAT_DIM)
        
    def forward(self, input_text, response_text, subcat_text):
        text_outputs = self.bert(**input_text)
        response_outputs = self.bert(**response_text)
        subcat_outputs = self.bert(**subcat_text)
        text_pooled_output = text_outputs[1]
        response_pooled_output = response_outputs[1]
        subcat_pooled_output = subcat_outputs[1]

        text_pooled_output = self.dropout(text_pooled_output)
        response_pooled_output = self.dropout(response_pooled_output)
        subcat_pooled_output = self.dropout(subcat_pooled_output)
        subcat_pooled_output = self.subcat_dense(subcat_pooled_output)
        pooled_output = self.fuse_dense(torch.cat([text_pooled_output, response_pooled_output, subcat_pooled_output], 1))
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return pooled_output, logits

# Encoder Decoder framework
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        # self.transformer = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.transformer = AutoModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_text):
        hidden_states = self.transformer(**input_text)[1]
        # hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits, hidden_states
    
class MT5Classifier(nn.Module):
    def __init__(self, num_labels):
        super(MT5Classifier, self).__init__()
        self.encoder = MT5EncoderModel.from_pretrained('google/mt5-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_text):
        hidden_states = self.encoder(**input_text, return_dict=True).last_hidden_state.mean(dim=1)
        # hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits, hidden_states

