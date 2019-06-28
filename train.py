import torch
import torch.autograd as autograd
import torch.nn as nn
import data_process
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
# from word_embedding import load_embeddings
from bilstm_crf import Bilstm_crf
from config import DefaultConfig
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def eval_model(model, x_valid,y_valid):
    valid_data = data_process.batch_iter(x_valid, y_valid, opt.batch_size, num_epochs=1)
    total_epoch_loss, total_epoch_acc, total_epoch_pre, total_epoch_recall, total_epoch_f1 = 0, 0, 0, 0, 0
    model.eval()
    for bat_num,valid_batch in enumerate(valid_data):
        text_lengths_val = [len(x) for x in valid_batch[0]]
        max_len_val=max(text_lengths_val)
        padded_text_val,text_lengths_val = data_process.pad_sequences(valid_batch[0])
        padded_tags_val,tag_lengths_val = data_process.pad_sequences(valid_batch[1])
        padded_text_val = torch.from_numpy(padded_text_val).long()
        target_val = torch.from_numpy(padded_tags_val).long()
        target_val = torch.autograd.Variable(target_val).long()
        if torch.cuda.is_available():
            padded_text_val = padded_text_val.cuda()
            target_val = target_val.cuda()
        prediction_val = model(padded_text_val, text_lengths_val)
        loss_val = model.loss(padded_text_val, text_lengths_val,target_val)
        """ evaluation :acc,precision,recall,f1"""
        num_corrects_val = (prediction_val.view(1, -1).data == target_val.view(1, -1).data).float().sum()

        acc_val = num_corrects_val / prediction_val.view(1, -1).size()[1]
        recall_val = data_process.get_score(prediction_val, target_val,'r')
        pre_val = data_process.get_score(prediction_val, target_val,'p')
        f1_val = data_process.get_score(prediction_val, target_val)
        # out_val = F.softmax(prediction_val, 1)
        """ 样本数据属于正例概率 """

        total_epoch_loss += loss_val.item()
        total_epoch_recall += recall_val
        total_epoch_acc += acc_val.item()
        total_epoch_pre += pre_val
        total_epoch_f1 += f1_val
        print(f"validation in batch:{bat_num+1}\n")

    model.train()
    return total_epoch_loss/(bat_num+1), total_epoch_acc/(bat_num+1), total_epoch_f1/(bat_num+1),total_epoch_pre/(bat_num+1),total_epoch_recall/(bat_num+1)

if __name__ == '__main__':
    opt = DefaultConfig()
    # 数据文件路径
    data_path = opt.data_path
    data = data_process.read_corpus(data_path)
    timestamp = str(int(time.time()))
    outdir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints", timestamp))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    vocab = data_process.load_vocab(opt.vocab_path)



    tag2label = {"O": 0,
                 "B-W": 1, "I-W": 2,
                 }
    tag2label[START_TAG] = len(tag2label)
    tag2label[STOP_TAG] = len(tag2label)
    print("映射word and tag to id")
    sentence = []
    for idx, data in enumerate(data):
        sent = list(data[0])
        sent = tokens_to_ids(sent,vocab)
        sentence.append(sent)
    tags = [[tag2label[tag] for tag in x[1]]for x in data]

    opt.parse({'vocab_size':len(words),
               'embedding_length': 300})

    model = Bilstm_crf(opt, tag2label)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    if torch.cuda.is_available():
        model = model.cuda()
    x_train, x_valid, y_train, y_valid = train_test_split(sentence,tags,test_size=0.2)
    train_data = data_process.batch_iter(x_train, y_train, opt.batch_size, num_epochs=opt.num_epochs)
    steps = 0
    min_delta = opt.min_delta
    best_loss = None
    best_acc, best_pre, best_f1 = 0, 0, 0
    epoch_length = len(x_train)//opt.batch_size
    model.train()
    for idx, batch in enumerate(train_data):
        text = batch[0]
        text_lengths = [len(x) for x in text]
        max_len = max(text_lengths)
        pad_token = words_dict['<pad>']
        # 将每个batch的text pad到这个batch里最长的长度
        padded_text = np.ones((opt.batch_size, max_len)) * pad_token
        for i, x_len in enumerate(text_lengths):
            sequence = text[i]
            padded_text[i, 0:x_len] = sequence[:x_len]
        padded_text,text_lengths = data_process.pad_sequences(text)
        padded_tags,tag_lengths = data_process.pad_sequences(batch[1])
        padded_text = torch.from_numpy(padded_text).long()

        target = torch.from_numpy(padded_tags).long()
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            padded_text = padded_text.cuda()
            target = target.cuda()
        # 每个batch都将梯度归零，不累计
        optim.zero_grad()
        prediction = model(padded_text, text_lengths)
        loss = model.loss(padded_text, text_lengths, target)
        loss.backward()
        # 所有tag预测对的个数
        num_corrects = (prediction.view(1, -1).data == target.view(1, -1).data).float().sum()
        acc = num_corrects / prediction.view(1,-1).size()[1]

        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        f1 = data_process.get_score(prediction, target)
        if steps % 200 == 0:
            """每epoch,保存已经训练的模型"""
            print(f'epoch: {steps // epoch_length}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .4f}, Training F1:{f1:.4f}')
            test_loss, test_acc,test_f1,test_pre,test_recall = eval_model(model, x_valid, y_valid)
            print(f'Val. Loss: {test_loss:.4f}, Val. Acc: {test_acc:.4f},Val.f1: {test_f1:.4f}，Val.pre:{test_pre:.4f},Val.recall:{test_recall:.4f}')
            # define early stopping
            if best_loss is None:
                best_loss = test_loss
            elif test_loss - best_loss > -min_delta:  # 降低的loss不够阈值
                if wait >= opt.patience:
                    print(f'Earlystopping in epoch ：{steps // epoch_length}')
                    break
                wait += 1
            else:
                wait = 1
                best_loss = test_loss
                best_acc = test_acc
                best_pre = test_pre
                best_f1 = test_f1
                best_recall =test_recall
                print(f'best loss: {test_loss:.3f}, best acc: {test_acc:.3f}')
                save_path = os.path.join(outdir, str(steps)) + '.pkl'
                save_dict = {"vocab_list": words, "state_dict": model.state_dict(), "optimizer": optim.state_dict(),
                             "epoch": steps // epoch_length}
                torch.save(save_dict, save_path)

    print(f'best loss: {best_loss:.3f}, best acc: {best_acc:.3f}, best precision: {best_pre:.3f},best_recall:{best_recall:.3f}')
