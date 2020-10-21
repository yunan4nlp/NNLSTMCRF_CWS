from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
import torch

class Tagger(object):
    def __init__(self, lstm_scorer, CRF, vocab, config):
        self.config = config
        self.lstm_scorer = lstm_scorer
        self.crf = CRF
        self.use_cuda = next(filter(lambda p: p.requires_grad, lstm_scorer.parameters())).is_cuda


    def train(self):
        self.lstm_scorer.train()
        self.crf.train()
        self.training = True

    def eval(self):
        self.lstm_scorer.eval()
        self.crf.eval()
        self.training = False


    def forward(self, batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask):
        if self.use_cuda:
            batch_chars = batch_chars.cuda()
            batch_bichars = batch_bichars.cuda()
            batch_extchars = batch_extchars.cuda()
            batch_extbichars = batch_extbichars.cuda()

            char_mask = char_mask.cuda()

        self.logit = self.lstm_scorer( batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask)

    def viterbi_decode(self, labels_mask):
        output = self.crf.viterbi_tags(self.logit, labels_mask)
        best_paths = []
        for path, score in output:
            best_paths.append(path)
        return best_paths

    def compute_loss(self, gold_labels, labels_mask):
        if self.use_cuda:
            gold_labels = gold_labels.cuda()
            labels_mask = labels_mask.cuda()
        b = gold_labels.size(0)
        crf_loss = -self.crf(self.logit, gold_labels, labels_mask) / b ##
        return crf_loss

    def compute_acc(self, gold_labels, labels_mask):
        b, seq_len = labels_mask.size()
        true_lengths = torch.sum(labels_mask, dim=1).numpy()
        pred_labels = self.logit.data.max(2)[1].cpu().numpy()
        gold_labels = gold_labels.cpu().numpy()
        correct = 0
        total = 0
        for idx in range(b):
            true_len = true_lengths[idx]
            total += true_len
            for idy in range(true_len):
                if pred_labels[idx][idy] == gold_labels[idx][idy]:
                    correct += 1
        return total, correct

