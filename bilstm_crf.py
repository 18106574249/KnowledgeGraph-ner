import torch
import torch.nn as nn
# from torchcrf import CRF

from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from crf import CRF


START_TAG = "<START>"
STOP_TAG = "<STOP>"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Bilstm_crf(nn.Module):
    def __init__(self, opt,tag2label):
        super(Bilstm_crf, self).__init__()

        self.embedding_length = opt.embedding_length
        self.hidden_size = opt.hidden_size
        self.output_size = len(tag2label)
        self.batch_size = opt.batch_size

        self.vocab_size = opt.vocab_size

        self.dropout = opt.dropout

        self.dropout_embed = nn.Dropout(opt.dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.dropout_embed = nn.Dropout(opt.dropout)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional = True,dropout=opt.dropout)

        if self.lstm.bidirectional:
            self.label = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.label = nn.Linear(self.hidden_size, self.output_size)
        self.crf = CRF(self.output_size)


    def loss(self, input_sentences, input_lengths,tags):

        feats = self._get_lstm_features(input_sentences,input_lengths)

        return self.crf.loss(feats,tags)

    def _get_lstm_features(self, input_sentences,input_length):
        input_sentences = input_sentences.permute(1, 0)
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        # input[batch_size,max_len,embeding_size]
        self.batch_size = input.shape[0]
        h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        input = nn.utils.rnn.pack_padded_sequence(input, input_length, batch_first=True, enforce_sorted=False)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        lstm_features = self.label(output)
        return lstm_features

    def forward(self, input_sentences, input_length,batch_size=None):
        # lstm_features [batch_size,max_seq_len,output_size]
        # tags [batch_size,max_seq_len]
        lstm_features = self._get_lstm_features(input_sentences, input_length)
        tags = self.crf(lstm_features)
        return tags