import time
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm

from data import Data
from model import ArgEncoder, Classifier

class ModelBuilder(object):
    def __init__(self, use_cuda, conf):
        self.cuda = use_cuda
        self.conf = conf
        self._pre_data()
        self._build_model()
    
    def _pre_data(self):
        print('pre data...')
        self.data = Data(self.cuda, self.conf)

    def _build_model(self):
        print('loading embedding...')
        if self.conf.corpus_splitting == 1:
            pre = './data/processed/lin/'
        elif self.conf.corpus_splitting == 2:
            pre = './data/processed/ji/'
        elif self.conf.corpus_splitting == 3:
            pre = './data/processed/l/'
        we = torch.load(pre+'we.pkl')
        char_table = None
        sub_table = None
        mem_mat = None
        if self.conf.need_char or self.conf.need_elmo:
            char_table = torch.load(pre+'char_table.pkl')
        if self.conf.need_sub:
            sub_table = torch.load(pre+'sub_table.pkl')
        if self.conf.need_mem_bank:
            mem_mat = torch.load(pre+'bank.pkl')
        print('building model...')
        self.encoder = ArgEncoder(self.conf, we, char_table, sub_table, mem_mat, self.cuda)
        self.classifier = Classifier(self.conf.clf_class_num, self.conf)
        if self.conf.is_mttrain:
            self.conn_classifier = Classifier(self.conf.conn_num, self.conf, True)
        if self.cuda:
            self.encoder.cuda()
            self.classifier.cuda()
            if self.conf.is_mttrain:
                self.conn_classifier.cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        para_filter = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        self.e_optimizer = torch.optim.Adagrad(para_filter(self.encoder), self.conf.lr, weight_decay=self.conf.l2_penalty)
        self.c_optimizer = torch.optim.Adagrad(para_filter(self.classifier), self.conf.lr, weight_decay=self.conf.l2_penalty)
        if self.conf.is_mttrain:
            self.con_optimizer = torch.optim.Adagrad(para_filter(self.conn_classifier), self.conf.lr, weight_decay=self.conf.l2_penalty)

    def _print_train(self, epoch, time, loss, acc, f1):
        print('-' * 80)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | loss: {:10.5f} | acc: {:5.2f}% | f1 {:5.2f}%'.format(
                epoch, time, loss, acc * 100, f1 * 100
            )
        )
        print('-' * 80)

    def _print_eval(self, task, loss, acc, f1):
        print(
            '| ' + task + ' loss {:10.5f} | acc {:5.2f}% | f1 {:5.2f}%'.format(loss, acc * 100, f1*100)
        )
        print('-' * 80)

    def _save_model(self, model, filename):
        torch.save(model.state_dict(), './weights/' + filename)

    def _load_model(self, model, filename):
        model.load_state_dict(torch.load('./weights/' + filename))

    def _update_mem(self):
        self.encoder.eval()
        self.classifier.eval()
        idx_list = []
        train_size = self.data.train_size
        for idx, a1, a2, sense, conn, ba1, ba2 in tqdm(self.data.train_loader, total=train_size // self.conf.batch_size + 1):
            if self.conf.four_or_eleven == 2:
                mask1 = (sense == self.conf.binclass)
                mask2 = (sense != self.conf.binclass)
                sense[mask1] = 1
                sense[mask2] = 0
            if self.cuda:
                idx, a1, a2, sense, conn, ba1, ba2 = idx.cuda(), a1.cuda(), a2.cuda(), sense.cuda(), conn.cuda(), ba1.cuda(), ba2.cuda()
            repr, mem_out, e = self.encoder(idx, a1, a2, ba1, ba2)
            self.encoder.bank.update(idx, repr)

            output = self.classifier([repr, mem_out])
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            mask = (output_sense == sense)
            correct_idx = torch.masked_select(idx, mask)
            idx_list.append(correct_idx)
        idx_list = torch.cat(idx_list, 0)
        self.encoder.bank.make_idx(idx_list)

    def _train_one(self, epoch):
        self.encoder.train()
        self.classifier.train()
        if self.conf.is_mttrain:
            self.conn_classifier.train()
        # if self.conf.need_mem_bank:
        #     if epoch % self.conf.mem_epoch == 0:
        #         self.classifier.only_mem = False
        #         # for optimizer in [self.e_optimizer, self.c_optimizer]:
        #         #     for param_group in optimizer.param_groups:
        #         #         param_group['lr'] = self.conf.lr
        #     else:
        #         self.classifier.only_mem = True
        #         # for optimizer in [self.e_optimizer, self.c_optimizer]:
        #         #     for param_group in optimizer.param_groups:
        #         #         param_group['lr'] = self.conf.lr
        total_loss = 0
        correct_n = 0
        train_size = self.data.train_size
        idx_list = []
        output_list = []
        gold_list = []
        for idx, a1, a2, sense, conn, ba1, ba2 in tqdm(self.data.train_loader, total=train_size//self.conf.batch_size+1):
            if self.conf.four_or_eleven == 2:
                mask1 = (sense == self.conf.binclass)
                mask2 = (sense != self.conf.binclass)
                sense[mask1] = 1
                sense[mask2] = 0
            if self.cuda:
                idx, a1, a2, sense, conn, ba1, ba2 = idx.cuda(), a1.cuda(), a2.cuda(), sense.cuda(), conn.cuda(), ba1.cuda(), ba2.cuda()
            repr, mem_out1, mem_out2, e = self.encoder(idx, a1, a2, ba1, ba2)
            if self.conf.need_mem_bank:
                self.encoder.bank.update(idx, repr)

            output = self.classifier([repr, mem_out1, mem_out2])
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            mask = (output_sense == sense)
            correct_n += torch.sum(mask.long()).item()
            output_list.append(output_sense)
            gold_list.append(sense)

            loss = self.criterion(output, sense)

            if self.conf.is_mttrain:
                conn_output = self.conn_classifier([repr, None])
                loss2 = self.criterion(conn_output, conn)
                loss = loss + loss2 * self.conf.lambda1

            self.e_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            if self.conf.is_mttrain:
                self.con_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.conf.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.conf.grad_clip)
            if self.conf.is_mttrain:
                torch.nn.utils.clip_grad_norm_(self.conn_classifier.parameters(), self.conf.grad_clip)
            self.e_optimizer.step()
            self.c_optimizer.step()
            if self.conf.is_mttrain:
                self.con_optimizer.step()
            
            total_loss += loss.item() * sense.size(0)

            correct_idx = torch.masked_select(idx, mask)
            idx_list.append(correct_idx)
        if self.conf.need_mem_bank:
            idx_list = torch.cat(idx_list, 0)
            self.encoder.bank.make_idx(idx_list)
        output_s = torch.cat(output_list)
        gold_s = torch.cat(gold_list)
        if self.conf.four_or_eleven == 2:
            f1 = f1_score(gold_s.cpu().numpy(), output_s.cpu().numpy(), average='binary')
        else:
            f1 = f1_score(gold_s.cpu().numpy(), output_s.cpu().numpy(), average='macro')

        return total_loss / train_size, correct_n / train_size, f1

    def _train(self, pre):
        best_dev_acc = 0
        for epoch in range(self.conf.epochs):            
            start_time = time.time()
            # self._update_mem()
            loss, acc, f1 = self._train_one(epoch)
            self._print_train(epoch, time.time()-start_time, loss, acc, f1)
            self.logwriter.add_scalar('loss/train_loss', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc', acc*100, epoch)

            dev_loss, dev_acc, dev_f1 = self._eval('dev')
            self._print_eval('dev', dev_loss, dev_acc, dev_f1)
            self.logwriter.add_scalar('loss/dev_loss', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc', dev_acc*100, epoch)
            self.logwriter.add_scalar('f1/dev_f1', dev_f1*100, epoch)

            test_loss, test_acc, test_f1 = self._eval('test')
            self._print_eval('test', test_loss, test_acc, test_f1)
            self.logwriter.add_scalar('loss/test_loss', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc', test_acc*100, epoch)
            self.logwriter.add_scalar('f1/test_f1', test_f1*100, epoch)

            # if test_acc >= best_dev_acc:
            #     best_dev_acc = test_acc
            #     self._save_model(self.encoder, 'eparams.pkl')
            #     self._save_model(self.classifier, 'cparams.pkl')
            #     print('params saved at epoch {}'.format(epoch))

    def train(self, pre):
        print('start training')
        self.logwriter = SummaryWriter(self.conf.logdir)
        self._train(pre)
        print('training done')

    def _eval(self, task):
        self.encoder.eval()
        self.classifier.eval()
        self.classifier.only_mem = False
        total_loss = 0
        correct_n = 0
        if task == 'dev':
            data = self.data.dev_loader
            n = self.data.dev_size
        elif task == 'test':
            data = self.data.test_loader
            n = self.data.test_size
        else:
            raise Exception('wrong eval task')
        output_list = []
        gold_list = []
        idx_list = []
        e_list = []
        with torch.no_grad():
            for idx, a1, a2, sense1, sense2, ba1, ba2 in data:
                if self.conf.four_or_eleven == 2:
                    mask1 = (sense1 == self.conf.binclass)
                    mask2 = (sense1 != self.conf.binclass)
                    sense1[mask1] = 1
                    sense1[mask2] = 0
                    mask0 = (sense2 == -1)
                    mask1 = (sense2 == self.conf.binclass)
                    mask2 = (sense2 != self.conf.binclass)
                    sense2[mask1] = 1
                    sense2[mask2] = 0  
                    sense2[mask0] = -1              
                if self.cuda:
                    idx, a1, a2, sense1, sense2, ba1, ba2 = idx.cuda(), a1.cuda(), a2.cuda(), sense1.cuda(), sense2.cuda(), ba1.cuda(), ba2.cuda()

                repr, mem_out1, mem_out2, e = self.encoder(idx, a1, a2, ba1, ba2)
                output = self.classifier([repr, mem_out1, mem_out2])
                _, output_sense = torch.max(output, 1)
                assert output_sense.size() == sense1.size()
                gold_sense = sense1
                mask = (output_sense == sense2)
                gold_sense[mask] = sense2[mask]
                tmp = (output_sense == gold_sense).long()
                correct_n += torch.sum(tmp).item()

                output_list.append(output_sense)
                gold_list.append(gold_sense)

                loss = self.criterion(output, gold_sense)
                total_loss += loss.item() * gold_sense.size(0)
                
            #     if task == 'test' and self.conf.need_mem_bank:
            #         idx_list.append(idx)
            #         e_list.append(e)
            # if task == 'test' and self.conf.need_mem_bank:
            #     idx_list = torch.cat(idx_list).cpu()
            #     e_list = torch.cat(e_list).cpu()
            #     print(idx_list[:10])
            #     for i in range(10):
            #         print(torch.topk(e_list[i], 3))
        
        output_s = torch.cat(output_list)
        gold_s = torch.cat(gold_list)
        if self.conf.four_or_eleven == 2:
            f1 = f1_score(gold_s.cpu().numpy(), output_s.cpu().numpy(), average='binary')
        else:
            f1 = f1_score(gold_s.cpu().numpy(), output_s.cpu().numpy(), average='macro')
        return total_loss / n, correct_n / n, f1

    def eval(self, pre):
        print('evaluating...')
        self._load_model(self.encoder, pre+'_eparams.pkl')
        self._load_model(self.classifier, pre+'_cparams.pkl')
        test_loss, test_acc, f1 = self._eval('test')
        self._print_eval('test', test_loss, test_acc, f1)
