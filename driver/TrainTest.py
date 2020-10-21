import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from data.Dataloader import *
from driver.Config import *
from data.Vocab import *
from modules.TaggerModel import LSTMscorer
from modules.Tagger import *
from data.Evaluate import *
import time
import itertools
import torch.nn as nn
import pickle




class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()

def train(train_data, dev_data, test_data, vocab, config, tagger):

    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             tagger.lstm_scorer.parameters(),
                             tagger.crf.parameters()
                         )
                         )

    model_optimizer = Optimizer(model_param, config)

    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))
    global_step = 0
    best_F = 0

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_correct,  overall_total = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, True):
            batch_gold_labels = label_variable(onebatch, vocab)

            batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask, label_mask = \
                data_variable(onebatch, vocab)
            tagger.train()

            tagger.forward(batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask)
            loss = tagger.compute_loss(batch_gold_labels, label_mask)

            total, correct = tagger.compute_acc(batch_gold_labels, label_mask)
            overall_total += total
            overall_correct += correct
            acc = overall_correct / overall_total
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            during_time = float(time.time() - start_time)
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)

                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                segment(dev_data, tagger, vocab, config, config.dev_file + '.' + str(global_step))
                dev_seg_eval = evaluate(config.dev_file, config.dev_file + '.' + str(global_step))

                print("Dev:")
                dev_seg_eval.print()

                segment(test_data, tagger, vocab, config, config.test_file + '.' + str(global_step))
                test_seg_eval = evaluate(config.test_file, config.test_file + '.' + str(global_step))

                print("Test:")
                test_seg_eval.print()

                dev_F = dev_seg_eval.getAccuracy()
                if best_F < dev_F:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_F, dev_F))
                    best_F = dev_F

                    if config.save_after >= 0 and iter >= config.save_after:
                        print("Save model")
                        tagger_model = {'lstm': tagger.lstm_scorer.state_dict(),
                                        'crf': tagger.crf.state_dict()}
                        torch.save(tagger_model, config.save_model_path + "." + str(global_step))


def segment(data, tagger, vocab, config, outputFile, split_str=' '):
    start = time.time()
    outf = open(outputFile, mode='w', encoding='utf8')
    for onebatch in data_iter(data, config.test_batch_size, False):
        b = len(onebatch)
        seg = False
        for idx in range(b):
            if len(onebatch[idx].chars) > 0:
                seg = True
                break
        if seg:
            batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask, label_mask = \
                data_variable(onebatch, vocab)
            tagger.eval()
            tagger.forward(batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask)
            best_paths = tagger.viterbi_decode(label_mask)

            labels = path2labels(best_paths, vocab)
            outputs = labels2output(onebatch, labels)
            for sent in outputs:
                outf.write(split_str.join(sent) + '\n')
        else:
            for idx in range(b):
                outf.write('\n')
    during_time = float(time.time() - start)
    outf.close()
    print("sentence num: %d,  segment time = %.2f " % (len(data), during_time))



if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseSegment')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    vocab = creatVocab(train_data, config.min_occur_count)

    pretrained_char, vocab._id2extchar, vocab._extchar2id = \
        load_pretrained_embs(config.pretrained_char_embeddings_file)
    pretrained_bichar, vocab._id2extbichar, vocab._extbichar2id = \
        load_pretrained_embs(config.pretrained_bichar_embeddings_file)

    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    start_a = time.time()

    get_gold_label(train_data)
    get_gold_label(dev_data)
    get_gold_label(test_data)

    print("train num: ", len(train_data))
    print("dev num: ", len(dev_data))
    print("test num: ", len(test_data))

    vocab.create_label(train_data)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    crf = CRF(num_tags=vocab.label_size,
              constraints=None,
              include_start_end_transitions=False)

    lstm_scorer = LSTMscorer(vocab, config)
    lstm_scorer.initial_by_pretrained(pretrained_char, pretrained_bichar)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True

        lstm_scorer = lstm_scorer.cuda()
        crf = crf.cuda()

    tagger = Tagger(lstm_scorer, crf, vocab, config)
    train(train_data, dev_data, test_data, vocab, config, tagger)

