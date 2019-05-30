import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class AttentionModel(torch.nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch
        output_size : 2
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing all words from external embeddings
        embedding_length : Embeddding dimension of word embeddings
        weights : Pre-trained word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = opt.batch_size
        self.output_size = opt.output_size
        self.hidden_size = opt.hidden_size
        self.vocab_size = opt.vocab_size
        self.embedding_length = opt.embedding_length
        self.dropout = opt.dropout

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.dropout_embed = nn.Dropout(opt.dropout)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional = True,dropout=opt.dropout)

        if self.lstm.bidirectional:
            self.label = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.label = nn.Linear(self.hidden_size, self.output_size)
    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.(batch_size,max_seq_len,hidden_size*2)
        final_state : Final time-step hidden state (h_n) of the LSTM (batch_size,hidden_size)

        ---------

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size*2)
                    attn_weights.size() = (batch_size, max_seq_len)
                    soft_attn_weights.size() = (batch_size, max_seq_len)
                    new_hidden_state.size() = (batch_size, hidden_size*2)

        """

        hidden = torch.cat((final_state[0], final_state[1]), 1)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences,input_length,batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, max_seq_len)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        final_output.shape = (batch_size, output_size)

        """
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_sentences = input_sentences.permute(1,0)
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        self.batch_size = input.shape[0]
        if torch.cuda.is_available():
            if batch_size is None:
                h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
                c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
            else:
                h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
                c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
        else:
            if batch_size is None:
                h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
                c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
            else:
                h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
                c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        input = nn.utils.rnn.pack_padded_sequence(input, input_length, batch_first=True,enforce_sorted=False)


        """
        output = (batch_size,max_seq_len,hidden_size*2)
        final_hidden_state.size() = (2, batch_size, hidden_size)
        att_output = (batch_size,hidden_size*2)
        """
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))  #

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits