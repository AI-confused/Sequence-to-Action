from transformers import BartConfig, BartModel, BertModel, BertPreTrainedModel, BertConfig, BertTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import torch
import torch.cuda.amp as amp
from torch.autograd import Variable


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    带有Softmax的label-smooth损失函数
    '''
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

class LabelSmoothCEV1(nn.Module):
    '''
    不带Softmax的label-smooth损失函数
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        # logs = self.log_softmax(logits)
        logs = torch.log(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class S2AModelBart(nn.Module):
    """
    基于腾讯的S2A论文的模型，Bart版本
    @config: dict
    """
    def __init__(self, config: dict):
        super(S2AModelBart, self).__init__()
        # Bart模块，替代了论文中的原生Transformer模块
        self.bart = BartModel.from_pretrained(config['model_path'])
        # 词表空间增加[BLK]
        self.bart.resize_token_embeddings(21129)
        # S2A模块的词嵌入向量层初始化
        self.s2a_embed = nn.Embedding(21128, config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        # S2A模块的动作分类层
        self.action1 = nn.Linear(config['d_model']*2, config['d_model']*2)
        self.action2 = nn.Linear(config['d_model']*2, 3)
        # Bart的Decoder解码器输出层
        self.out = nn.Linear(config['d_model'], config['trg_vocab'])
        # 两部分损失组合的权重
        self.alpha = config['alpha']


    def forward(self, encoder_input, decoder_input, encoder_input_mask, decoder_input_mask, x_hat, decoder_output):
        """
        @encoder_input: Bart的Encoder输入 (batch, src_len)
        @decoder_input: 
        """
        assert x_hat.size(1) == decoder_input.size(1) == decoder_output.size(1)
        # 获取Bart模块的token生成结果 (batch, trg_len, hidden_dim)
        d_outputs = self.bart(input_ids=encoder_input, attention_mask=encoder_input_mask,
                    decoder_input_ids=decoder_input, decoder_attention_mask=decoder_input_mask)
        d_output = self.dropout(d_outputs['last_hidden_state'])
        a_output = F.softmax(self.action2(self.action1(torch.cat((d_output, self.s2a_embed(x_hat)), -1))), dim=-1)
        d_output = self.out(d_output)
        if decoder_output is not None:
            decoder_output = decoder_output.contiguous().view(-1)
            # 训练阶段
            loss_s2s = LabelSmoothSoftmaxCEV1(ignore_index=-1)(d_output.view(-1, d_output.size(-1)), decoder_output)
            
            d_a_output = d_output.clone()
            # 0-S, 1-C, 2-G
            # 将S和C概率设置为0
            d_a_output.data[:,:,21128] = 0.0 # in-place
            d_a_output.data = d_a_output.data.scatter(2, x_hat.unsqueeze(-1), 0.0)
            # 概率归一化
            d_a_output = F.softmax(d_a_output, dim=-1)
            # 将Gen的概率乘上decoder的输出
            d_a_output = torch.mul(d_a_output, a_output[:,:,2].unsqueeze(-1))
            # 将Skip的概率赋值给decoder的输出在[BLK]位置的概率
            d_a_output = d_a_output.scatter(2, torch.mul(torch.ones_like(x_hat), 21128).unsqueeze(-1), a_output[:, :, 0].unsqueeze(-1))
            # # 将Copy的概率赋值给decoder的输出在x_hat(i)位置的概率
            d_a_output = d_a_output.scatter(2, x_hat.unsqueeze(-1), a_output[:, :, 1].unsqueeze(-1))
            
            loss_s2a = LabelSmoothCEV1(ignore_index=-1)(d_a_output.view(-1, d_a_output.size(-1)), decoder_output)

            loss = (1-self.alpha)*loss_s2a + self.alpha*loss_s2s
            return loss
        else:
            # 推理阶段
            return None


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad != None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad != None:
                param.grad = self.grad_backup[name]