from modules.Layer import *

class LSTMscorer(nn.Module):
    def __init__(self, vocab, config):
        super(LSTMscorer, self).__init__()
        self.config = config
        self.char_embed = nn.Embedding(vocab.char_size, config.char_dims, padding_idx=0)
        self.extchar_embed = nn.Embedding(vocab.extchar_size, config.char_dims, padding_idx=0)

        self.bichar_embed = nn.Embedding(vocab.bichar_size, config.bichar_dims, padding_idx=0)
        self.extbichar_embed = nn.Embedding(vocab.extbichar_size, config.bichar_dims, padding_idx=0)

        char_init = np.zeros((vocab.char_size, config.char_dims), dtype=np.float32)
        self.char_embed.weight.data.copy_(torch.from_numpy(char_init))

        bichar_init = np.zeros((vocab.bichar_size, config.bichar_dims), dtype=np.float32)
        self.bichar_embed.weight.data.copy_(torch.from_numpy(bichar_init))

        self.extchar_embed.weight.requires_grad = False

        self.extbichar_embed.weight.requires_grad = False

        self.char_linear = nn.Linear(in_features=config.char_dims * 2 + config.bichar_dims * 2,
                                     out_features=config.hidden_size,
                                     bias=True)

        self.lstm = MyLSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.score = nn.Linear(in_features=config.lstm_hiddens * 2,
                               out_features=vocab.label_size,
                               bias=False)

        torch.nn.init.kaiming_uniform_(self.score.weight)

    def initial_by_pretrained(self, pretrained_char, prtrained_bichar):
        self.extchar_embed.weight.data.copy_(torch.from_numpy(pretrained_char))
        self.extbichar_embed.weight.data.copy_(torch.from_numpy(prtrained_bichar))


    def forward(self,  batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask):
        chars_emb = self.char_embed(batch_chars)
        extchars_emb = self.extchar_embed(batch_extchars)
        bichars_emb = self.bichar_embed(batch_bichars)
        extbichars_emb = self.extbichar_embed(batch_extbichars)

        if self.training:
            chars_emb = drop_input_independent(chars_emb, self.config.dropout_emb)
            extchars_emb = drop_input_independent(extchars_emb, self.config.dropout_emb)
            bichars_emb = drop_input_independent(bichars_emb, self.config.dropout_emb)
            extbichars_emb = drop_input_independent(extbichars_emb, self.config.dropout_emb)

        char_represents = torch.cat([chars_emb, extchars_emb, bichars_emb, extbichars_emb], -1)


        char_hidden = torch.tanh(self.char_linear(char_represents))
        lstm_hidden, _ = self.lstm(char_hidden, char_mask, None)
        lstm_hidden= lstm_hidden.transpose(1, 0)
        score = self.score(lstm_hidden)
        return score



