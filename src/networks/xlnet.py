import torch.nn as nn

from transformers import XLNetModel, XLNetForSequenceClassification, XLNetTokenizer

class XLNET(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self, pretrained_model_name='xlnet-base-cased', cache_dir='../data/xlnet_cache'):
        super().__init__()

        # Check if choice of pretrained model is valid
        assert pretrained_model_name in ('xlnet-base-cased')

        # Load pre-trained BERT model
        # self.bert = BertModel.from_pretrained(pretrained_model_name=pretrained_model_name, cache_dir=cache_dir)
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)

        self.embedding = self.xlnet.embeddings
        self.embedding_size = self.embedding.word_embeddings.embedding_dim

        # Remove BERT model parameters from optimization
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        self.xlnet.eval()  # make sure bert is in eval() mode
        hidden, _ = self.xlnet(x.transpose(0, 1), output_all_encoded_layers=False)  # output only last layer
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        # Change to hidden.shape = (sentence_length, batch_size, hidden_size) align output with word embeddings
        hidden = hidden.transpose(0, 1)

        return hidden
