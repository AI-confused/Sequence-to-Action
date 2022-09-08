import os
import random
import logging
import tqdm
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import math
import torch.nn.functional as F
from transformers import BertConfig, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from easy_task.base_module import *
from model import S2AModelBart, PGD, FGM
from correct_result import GrammarCorrectResult
from torch.autograd import Variable



class GrammarCorrectTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=False, load_dev: bool=False, load_test: bool=False):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(GrammarCorrectTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

        # prepare model
        self.prepare_task_model()

        # load dataset
        self.load_data(load_train, load_dev, load_test)

        # prepare optim
        if load_train:
            self.prepare_optimizer()

        self._decorate_model()

        # best score and output result(custom)
        self.best_dev_score = 0.0
        self.best_dev_epoch = 0
        self.output_result = {'result_type': '', 'task_config': self.setting.__dict__, 'result': []}


    def prepare_task_model(self):
        """Prepare ner task model(custom).

        Can be overwriten.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.setting.model_name)
        # 添加[BLK]
        special_tokens_dict = {'additional_special_tokens': ['[BLK]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.cls_id, self.sep_id = 101, 102
        # 模型的配置字典
        self.config = {
            'src_vocab': len(self.tokenizer.vocab)+1,
            'trg_vocab': len(self.tokenizer.vocab)+1,
            'dropout': self.setting.dropout,
            'model_path': self.setting.model_name,
            'alpha': self.setting.alpha,
            }
        if 'base' in self.setting.model_name:
            self.config['d_model'] = 768
            self.model = S2AModelBart(config=self.config)
            # S2A模块的词嵌入层加载bart-base的权重初始化
            self.model.s2a_embed.load_state_dict(torch.load('/home/liyunliang/CGED_Task/dataset/bart_base_embedding.pkl'))
        else:
            self.config['d_model'] = 1024
            self.model = S2AModelBart(config=self.config)
            # S2A模块的词嵌入层加载bart-large的权重初始化
            self.model.s2a_embed.load_state_dict(torch.load('/home/liyunliang/CGED_Task/dataset/bart_large_embedding.pkl'))


    def prepare_optimizer(self):
        """Prepare task optimizer(custom).

        Can be overwriten.
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.setting.learning_rate))

        num_train_steps = int(len(self.train_features) / self.setting.train_batch_size / self.setting.gradient_accumulation_steps * self.setting.num_train_epochs)

        # scheduler
        if hasattr(self.setting, 'scheduler') and self.setting.scheduler=='linear':
            self.logger.info('================ do scheduler linear ================')
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.setting.warmup_portion*num_train_steps), num_training_steps=num_train_steps
            )
        elif hasattr(self.setting, 'scheduler') and self.setting.scheduler=='cosine':
            self.logger.info('================ do scheduler cosine ================')
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.setting.warmup_portion*num_train_steps), num_training_steps=num_train_steps, num_cycles=0.5
            )
        else:
            self.scheduler = None

        # 对抗训练
        if hasattr(self.setting, 'adverse_train') and self.setting.adverse_train=='fgm':
            self.logger.info('================ do adverse train fgm ================')
            self.adverse_attack = FGM(self.model)
        elif hasattr(self.setting, 'adverse_train') and self.setting.adverse_train=='pgd':
            self.logger.info('================ do adverse train pgd ================')
            self.adverse_attack = PGD(self.model)
        else:
            self.adverse_attack = None


    def prepare_result_class(self):
        """Prepare result calculate class(custom).

        Can be overwriten.
        """
        self.result = GrammarCorrectResult(task_name=self.setting.task_name)


    def load_examples_features(self, data_type: str, file_name: str) -> tuple:
        """Load examples, features and dataset(custom).

        Can be overwriten, but with the same input parameters and output type.
        
        @data_type: train or dev or test
        @file_name: dataset file name
        """
        cached_features_file0 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'examples'))
        cached_features_file1 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'features'))

        if not self.setting.over_write_cache and os.path.exists(cached_features_file0) and os.path.exists(cached_features_file1):
            examples = torch.load(cached_features_file0)
            features = torch.load(cached_features_file1)
        else:
            examples = self.read_examples(os.path.join(self.setting.data_dir, file_name), data_type=data_type, percent=self.setting.percent)
            # 由于数据量太大，因此不保存
            # torch.save(examples, cached_features_file0)
            features = self.convert_examples_to_features(examples,
                                                        tokenizer=self.tokenizer,
                                                        max_seq_len=self.setting.max_seq_len,
                                                        data_type=data_type)
            # 由于数据量太大，因此不保存
            # torch.save(features, cached_features_file1)
        dataset = TextDataset(features)
        return (examples, features, dataset, features[0].max_seq_len)


    def read_examples(self, input_file: str, data_type: str, percent: float=1.0) -> list:
        """Read data from a data file and generate list of InputExamples(custom).

        Can be overwriten, but with the same input parameters and output type.

        @input_file: data input file abs dir
        @percent: percent of reading samples
        """
        examples=[]

        data = open(input_file, 'r').readlines()
        
        # 小批量测试
        if percent != 1.0:
            data = random.sample(data, int(len(data)*percent))

        for line in tqdm.tqdm(data, total=len(data), desc='read examples'):
            line = json.loads(line)

            if line['error'] == []:
                detect_label = 0
                identification_label = []
            else:
                detect_label = 1

                identification_label = []
                for item in line['error']:
                    identification_label.append(item['type'])

            examples.append(InputExample(
                                file_name=line['file_name'],
                                doc_id=line['id'],
                                text=line['text'],
                                correct=line['correct'],
                                detect_label=detect_label,
                                identification_label=identification_label,
                                position_label=line['error']
            ))            

        return examples


    def convert_examples_to_features(self, examples: list, tokenizer: BertTokenizer, max_seq_len: int, **kwargs) -> list:
        """Process the InputExamples into InputFeatures that can be fed into the model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples: list of InputExamples
        @tokenizer: class BertTokenizer or its inherited classes
        @max_seq_len: max length of tokenized text
        """
        features = []
        if kwargs['data_type'] == 'train':
            for example in tqdm.tqdm(examples, total=len(examples), desc='convert features'):
                if example.correct == '':
                    continue
                # 构造z&a
                flag = 0
                labels = sorted(example.position_label, key=lambda x: x['start'])
                z, a = '', ''
                last_point = 0
                cor_last_point = 0
                for label in labels:
                    if label['start'] < last_point:
                        continue
                    try:
                        assert label['end'] - label['start'] >= 1
                    except:
                        # 语病标签有误
                        flag = 1
                        # print('语病标签有误: ', example.text, ' / ', example.correct)
                        break

                    z += example.text[last_point:label['start']]
                    a += 'C'*(label['start']-last_point)
                    cor_last_point += label['start']-last_point

                    try:
                        if label['type'] == 'M':
                            z += (label['correct']+example.text[label['start']])
                            a += ('G'*len(label['correct'])+'C')
                            last_point = label['start']+1
                            cor_last_point += (1+len(label['correct']))
                        elif label['type'] == 'R':
                            z += example.text[label['start']:label['end']]
                            a += 'S'*(label['end']-label['start'])
                            last_point = label['end']
                        elif label['type'] == 'S':
                            z += (example.text[label['start']:label['end']] + label['correct'])
                            a += ('S'*(label['end']-label['start']) + 'G'*len(label['correct']))
                            last_point = label['end']
                            cor_last_point += len(label['correct'])
                        else:
                            up, down = example.text[label['start']:label['end']], example.correct[cor_last_point:cor_last_point+label['end']-label['start']]
                            assert len(up) == len(down)
                            for i in range(len(up)-1, -1, -1):
                                if down[i] == up[0]:
                                    break
                            z += (down + up[len(down)-i:])
                            a += ('G'*i + 'C'*(len(down)-i) + 'S'*i)
                            last_point = label['end']
                            cor_last_point += len(down)
                    except:
                        # 最后一个字符缺失
                        # print('最后一个字符缺失: ', example.text, ' / ', example.correct)
                        continue
                if flag:
                    continue
                
                z += example.text[last_point:]
                a += 'C'*(len(example.text)-last_point)
                assert len(z) == len(a)
                # 构造模型的输入
                try:
                    x_hat, y_hat = [], []
                    j = 0
                    for i in range(len(z)):
                        if a[i] == 'C':
                            x_hat.append(z[i])
                            j += 1
                            y_hat.append(z[i])
                        elif a[i] == 'S':
                            x_hat.append(z[i])
                            j += 1
                            y_hat.append('[BLK]')
                        else:
                            x_hat.append(example.text[j])
                            y_hat.append(z[i])
                    assert len(x_hat) == len(y_hat)
                except:
                    # 'G'位于文本末尾
                    # print("'G'位于文本末尾: ", example.text, ' / ', example.correct)
                    continue

                z += '[SEP]'
                a += 'C'

                encoder_input = self.tokenizer.encode(
                    example.text,
                    max_length=self.setting.max_seq_len,
                    add_special_tokens=True
                )

                s2a_input = self.tokenizer.encode(
                    x_hat,
                    max_length=self.setting.max_seq_len-1,
                    add_special_tokens=False
                ) + [self.sep_id]

                decoder_output = self.tokenizer.encode(
                    y_hat,
                    max_length=self.setting.max_seq_len-1,
                    add_special_tokens=False
                ) + [self.sep_id]

                decoder_input = [self.cls_id]
                for index, id in enumerate(decoder_output[:-1]):
                    if id == 21128:
                        decoder_input.append(decoder_input[index-1])
                    else:
                        decoder_input.append(decoder_output[index])
                try:
                    assert len(decoder_input) == len(decoder_output) == len(a)
                except:
                    continue

                features.append(
                    InputFeature(
                        doc_id=example.doc_id,
                        ori=example.text,
                        cor=example.correct,
                        z=z,
                        a=a,
                        x_hat=x_hat,
                        y_hat=y_hat,
                        encoder_input=encoder_input,
                        s2a_input=s2a_input,
                        decoder_input=decoder_input,
                        decoder_output=decoder_output,
                        max_seq_len=max_seq_len,
                    )
                )
        elif kwargs['data_type'] in ['dev', 'test']: 
            for example in tqdm.tqdm(examples, total=len(examples), desc='convert features'):
                encoder_input = self.tokenizer.encode(
                    example.text,
                    max_length=self.setting.max_seq_len,
                    add_special_tokens=True
                )

                features.append(
                    InputFeature(
                        doc_id=example.doc_id,
                        file_name=example.file_name,
                        text=example.text,
                        detect_label=example.detect_label,
                        identification_label=example.identification_label,
                        position_label=example.position_label,
                        encoder_input=encoder_input,
                        max_seq_len=max_seq_len,
                    )
                )
        else:
            raise Exception('Error')
        
        return features


    def train(self, resume_base_epoch: int=None, resume_model_path: str=None):
        """Task level train func.

        @resume_base_epoch(int): start training epoch
        """
        self.logger.info('=' * 20 + 'Start Training {}'.format(self.setting.task_name) + '=' * 20)

        # resume model when restarting
        if resume_base_epoch is not None and resume_model_path is not None:
            raise ValueError('resume_base_epoch and resume_model_path can not be together!')
        elif resume_model_path is not None:
            self.logger.info('Training starts from other model: {}'.format(resume_model_path))
            self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=True)
            resume_base_epoch = 0
        else:
            if resume_base_epoch is None:
                if self.setting.resume_latest_cpt:
                    resume_base_epoch = self.get_latest_cpt_epoch()
                else:
                    resume_base_epoch = 0

            # resume cpt if possible
            if resume_base_epoch > 0:
                self.logger.info('Training starts from epoch {}'.format(resume_base_epoch))
                self.resume_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, resume_base_epoch, resume_base_epoch, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed), resume_model=True, resume_optimizer=True)
            else:
                self.logger.info('Training starts from scratch')

        # prepare data loader
        self.train_dataloader = self._prepare_data_loader(self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train)

        # do base train
        self._base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_epoch: {} - best_dev_score: {}'.format(self.best_dev_epoch, self.best_dev_score))

        # write output results
        self.write_results()

    
    def eval(self, global_step):
        """Task level eval func.

        @epoch(int): eval epoch
        """        
        data_type = 'dev'
        features = self.dev_features
        examples = self.dev_examples
        dataset = self.dev_dataset

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do base eval
        self._base_eval(global_step, data_type, examples, features)

        # calculate result score
        score = self.result.get_score()
        self.logger.info(json.dumps(score, indent=2, ensure_ascii=False))

        # return bad case in train-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='dev_bad_case_pos', items=self.result.bad_case_pos, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_iden', items=self.result.bad_case_iden, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_det', items=self.result.bad_case_det, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_FPR', items=self.result.bad_case_FPR, data_type='dev', epoch=global_step, file_type='excel')

        # return all result
        self.return_selected_case(type_='eval_prediction', items=self.result.all_result, data_type=data_type, epoch=global_step, file_type='excel')
        
        # save each epoch result
        self.output_result['result'].append('data_type: {} - checkpoint: {} - train_loss: {} - epoch_score: {}'\
                                            .format(data_type, global_step, self.train_loss, json.dumps(score, indent=2, ensure_ascii=False)))

        # save best model with specific standard(custom)
        if data_type == 'dev' and score['detect_f1']+score['iden_f1']+score['posi_f1']-score['FPR'] > self.best_dev_score:
            self.best_dev_step = global_step
            self.best_dev_score = score['detect_f1']+score['iden_f1']+score['posi_f1']-score['FPR']
            self.logger.info('saving best dev model [{}]...'.format(self.best_dev_score))
            self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, data_type, 0, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))
            
        save_cpt_file = '{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)
        if self.setting.save_cpt_flag == 1 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save last epoch
            last_checkpoint = self.get_latest_cpt_epoch()
            if last_checkpoint != 0:
                # delete lastest epoch model and store this epoch
                delete_cpt_file = '{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, last_checkpoint, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)

                if os.path.exists(os.path.join(self.setting.model_dir, delete_cpt_file)):
                    os.remove(os.path.join(self.setting.model_dir, delete_cpt_file))
                    self.logger.info('remove model {}'.format(delete_cpt_file))
                else:
                    self.logger.info("{} does not exist".format(delete_cpt_file))

            self.logger.info('saving latest epoch model...')
            self.save_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))

        elif self.setting.save_cpt_flag == 2 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save each epoch
            self.logger.info('saving checkpoint {} model...'.format(global_step))
            self.save_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))


    def nopeak_mask(self, size, batch=1):
        """上三角mask矩阵
        """
        np_mask = np.triu(np.ones((batch, size, size)), k=1).astype('uint8')
        np_mask =  Variable(torch.from_numpy(np_mask) == 0)
        return np_mask


    def create_masks(self, src, trg):
        # encoder对pad部分进行mask
        src_mask = (src != self.tokenizer.pad_token_id)

        if trg is not None:
            # decoder对pad进行mask
            trg_mask = (trg != self.tokenizer.pad_token_id)
        else:
            trg_mask = None
        return src_mask, trg_mask


    def custom_collate_fn_train(self, features: list) -> list:
        """Convert batch training examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        encoder_input = pad_sequence([torch.tensor(feature.encoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        s2a_input = pad_sequence([torch.tensor(feature.s2a_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_input = pad_sequence([torch.tensor(feature.decoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_output = pad_sequence([torch.tensor(feature.decoder_output, dtype=torch.long) for feature in features], batch_first=True, padding_value=-1)
        ori_mask, cor_mask = self.create_masks(encoder_input, decoder_input)

        return [encoder_input, s2a_input, decoder_input, decoder_output, ori_mask, cor_mask, features]


    def custom_collate_fn_eval(self, features: list) -> list:
        """Convert batch eval examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        encoder_input = pad_sequence([torch.tensor(feature.encoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ori_mask = (encoder_input != self.tokenizer.pad_token_id)

        return [encoder_input, ori_mask, features]


    def resume_test_at(self, resume_model_path: str, **kwargs):
        """Resume checkpoint and do test(custom).

        Can be overwriten, but with the same input parameters.
        
        @resume_model_path: do test model path
        """
        # extract kwargs
        header = kwargs.pop("header", None)

        self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=False)

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(self.test_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do test
        self._base_eval(0, 'test', self.test_examples, self.test_features)

        # calculate result score
        score = self.result.get_score()
        self.logger.info(json.dumps(score, indent=2, ensure_ascii=False))

        # return bad case in test-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='dev_bad_case_pos', items=self.result.bad_case_pos, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_iden', items=self.result.bad_case_iden, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_det', items=self.result.bad_case_det, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_FPR', items=self.result.bad_case_FPR, data_type='dev', epoch=0, file_type='excel')

        # return all result
        self.return_selected_case(type_='test_prediction', items=self.result.all_result, data_type='test', epoch=0, file_type='excel')
    

    def get_result_on_batch(self, batch: tuple):
        """Return batch output logits during eval model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        encoder_input, ori_mask, features = batch
        predicts = self.beam_search(encoder_input, ori_mask, features[0])
        return predicts, features


    def get_loss_on_batch(self, batch: tuple):
        """Return batch loss during training model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        encoder_input, s2a_input, decoder_input, decoder_output, ori_mask, cor_mask, features = batch
        loss = self.model(encoder_input, decoder_input, ori_mask, cor_mask, s2a_input, decoder_output)
        return loss


    def init_vars(self, src, src_mask):  
        init_tok = self.cls_id
        # src_mask = (src != tokenizer.pad_id()).unsqueeze(-2)
        if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
            e_output = self.model.module.bart.encoder(src, src_mask)
        else:
            e_output = self.model.bart.encoder(src, src_mask)
        
        outputs = torch.LongTensor([[init_tok] for _ in range(src.size(0))]).to(self.device)
        
        # trg_mask = self.nopeak_mask(1, batch=src.size(0))
        # trg_mask = trg_mask.to(self.device) # 添加的代码
        trg_mask = (outputs != self.tokenizer.pad_token_id)

        # 取出src中每个样本的首字符输入S2A module
        index_x = torch.arange(src.size(0)).view(src.size(0))
        index_y = torch.ones(src.size(0), dtype=torch.long)
        if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
            d_outputs = self.model.module.bart.decoder(input_ids=outputs, attention_mask=trg_mask,
                                                encoder_hidden_states=e_output['last_hidden_state'], encoder_attention_mask=src_mask)
            # S2A Module
            a_output = F.softmax(self.model.module.action2(self.model.module.action1(torch.cat((d_outputs.last_hidden_state, self.model.module.s2a_embed(src[index_x, index_y].unsqueeze(-1))), -1))), dim=-1)
            d_output = self.model.module.out(d_outputs.last_hidden_state)
        else:
            d_outputs = self.model.bart.decoder(input_ids=outputs, attention_mask=trg_mask,
                                        encoder_hidden_states=e_output['last_hidden_state'], encoder_attention_mask=src_mask)
            # S2A Module
            a_output = F.softmax(self.model.action2(self.model.action1(torch.cat((d_outputs.last_hidden_state, self.model.s2a_embed(src[index_x, index_y].unsqueeze(-1))), -1))), dim=-1)
            d_output = self.model.out(d_outputs.last_hidden_state)
        # 0-S, 1-C, 2-G
        
        ########### 根据融合后的词表输出解码 #################
        # 将S和C概率设置为0
        # d_output[:,:,21128] = 0
        # d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1), 0)
        # # 概率归一化
        # d_output = F.softmax(d_output, dim=-1)
        # # 将Gen的概率乘上decoder的输出
        # d_output *= a_output[:,:,2].unsqueeze(-1)
        # # 将Skip的概率赋值给decoder的输出在[BLK]位置的概率
        # d_output[:,:,21128] = a_output[:, :, 0]
        # # 将Copy的概率赋值给decoder的输出在x_hat(i)位置的概率
        # d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1), a_output[:, :, 1].unsqueeze(-1)) # src[index_x, index_y] == src[:, 0] 
        # # 获取结合S2A模块的在词表空间的输出
        # d_output_argmax = torch.argmax(d_output, dim=-1).squeeze(-1) # (batch,)
        # # 输出结果为Copy
        # tag_C = d_output_argmax.cpu() == src[index_x, index_y].cpu()
        # # 输出结果为Skip
        # tag_S = d_output_argmax.cpu() == torch.tensor([21128]*src.size(0))
        # tag_CS = tag_C | tag_S
        # x_hat_delta = torch.zeros(src.size(0), dtype=torch.long).masked_fill(tag_CS, 1)
        # index_y += x_hat_delta
        ########### 根据融合后的词表输出解码 #################

        ########### 根据Action输出解码 #################
        # 将S和C概率设置为0
        d_output[:,:,21128] = 0
        d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1), 0)
        # 概率归一化
        d_output = F.softmax(d_output, dim=-1)
        # 推理时根据3个动作的概率
        a_output_argmax = torch.argmax(a_output, dim=-1).squeeze(-1) # (batch,)
        # 输出结果为Copy
        tag_C = a_output_argmax.cpu() == torch.ones(src.size(0), dtype=torch.long)
        # 输出结果为Skip
        tag_S = a_output_argmax.cpu() == torch.zeros(src.size(0), dtype=torch.long)
        tag_CS = tag_C | tag_S
        x_hat_delta = torch.zeros(src.size(0), dtype=torch.long).masked_fill(tag_CS, 1)
        
        # 忽略S和C，计算所有batch的
        probs, ix = d_output[:, -1].data.topk(self.setting.beam_k)
        # log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
        log_scores = torch.Tensor([[math.log(prob) for prob in probs.data[i]] for i in range(probs.size(0))])
        
        outputs = torch.zeros(src.size(0), self.setting.beam_k, self.setting.max_seq_len).long().to(self.device)
        # if opt.device == 0:
        #     outputs = outputs.cuda()
        outputs[:, :, 0] = init_tok
        # 先给所有batch赋上Generate的token输出
        outputs[:, :, 1] = ix

        # 给标记为Skip的位置赋上[BLK]的输出
        new_tag_S = torch.zeros_like(outputs, dtype=torch.bool)
        new_tag_S[:, :, 1] = tag_S.unsqueeze(-1)
        outputs = outputs.masked_fill(new_tag_S, 21128)

        # 给标记为Copy的位置赋上x
        for j in range(src.size(0)):
            if tag_C[j] == True:
                outputs[j][0][1] = src[index_x, index_y][j]

        # 最后更新index_y
        index_y += x_hat_delta
        ########### 根据Action输出解码 #################
        
        # e_outputs = torch.zeros(src.size(0), self.setting.beam_k, e_output.size(-2), e_output.size(-1)).to(self.device)
        # if opt.device == 0:
        #     e_outputs = e_outputs.cuda()
        e_outputs = e_output['last_hidden_state'].unsqueeze(1).expand(src.size(0), self.setting.beam_k, e_output['last_hidden_state'].size(-2), e_output['last_hidden_state'].size(-1))
        # beam_3
        # for i in range(e_outputs.size(0)):
        #     # for j in range(e_outputs.size(1)):
        #     if not e_outputs[i][0].cpu().numpy().tolist()==e_outputs[i][1].cpu().numpy().tolist()==e_outputs[i][2].cpu().numpy().tolist():
        #         raise Exception('error')

        
        return outputs, e_outputs, log_scores, index_y
        

    def k_best_outputs(self, outputs, out, log_scores, i, k):
        out = out.contiguous().view(out.shape[0]//k, k, out.size(-2), out.size(-1)) # (batch, beam_k, seq_len, vocab_size)
        probs, ix = out[:, :, -1].data.topk(k) # 取seq最后字符输出 (batch, seqlen, topk)
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(probs.size(0), k, -1) + log_scores.unsqueeze(-1).expand(log_scores.size(0), log_scores.size(1), k)
        k_probs, k_ix = log_probs.view(log_probs.size(0), -1).topk(k)
        
        row = k_ix // k
        col = k_ix % k

        outputs = outputs.contiguous().view(outputs.shape[0]//k, k, outputs.shape[-1])
        for b in range(outputs.size(0)):
            outputs[b][:, :i] = outputs[b][row[b], :i]
        # outputs[:, :i] = outputs[row, :i]
            outputs[b][:, i] = ix[b][row[b], col[b]]

        # log_scores = k_probs.unsqueeze(0)
        
        return outputs, log_scores


    def beam_search(self, src, src_mask, feature):
        batch_outputs = ['' for _ in range(src.shape[0])]
        outputs, e_outputs, log_scores, index_y = self.init_vars(src, src_mask)

        e_outputs = e_outputs.contiguous().view(e_outputs.size(0)*self.setting.beam_k, e_outputs.size(-2), e_outputs.size(-1))
        src_mask = src_mask.unsqueeze(-2).expand(src_mask.size(0), self.setting.beam_k, src_mask.size(-1))
        src_mask = src_mask.contiguous().view(src_mask.size(0)*self.setting.beam_k, src_mask.size(-1))
        ind = None
        # x_hat的索引矩阵
        index_x = torch.arange(src.size(0)).view(src.size(0))
        for i in range(2, feature.max_seq_len):
            trg_mask = (outputs != self.tokenizer.pad_token_id)
            # 降维度
            trg_mask = trg_mask.view(trg_mask.size(0)*self.setting.beam_k, trg_mask.size(-1))
            outputs = outputs.view(outputs.size(0)*self.setting.beam_k, outputs.size(-1))

            if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
                d_outputs = self.model.module.bart.decoder(input_ids=outputs[:, :i], attention_mask=trg_mask[:, :i],
                                                encoder_hidden_states=e_outputs, encoder_attention_mask=src_mask)
                # S2A Module
                a_output = F.softmax(self.model.module.action2(self.model.module.action1(torch.cat((d_outputs.last_hidden_state[:, -1].unsqueeze(1), self.model.module.s2a_embed(src[index_x, index_y].unsqueeze(-1))), -1))), dim=-1)
                d_output = self.model.module.out(d_outputs.last_hidden_state)
            else:
                d_outputs = self.model.bart.decoder(input_ids=outputs[:, :i], attention_mask=trg_mask[:, :i],
                                                encoder_hidden_states=e_outputs, encoder_attention_mask=src_mask)
                # S2A Module
                a_output = F.softmax(self.model.action2(self.model.action1(torch.cat((d_outputs.last_hidden_state[:, -1].unsqueeze(1), self.model.s2a_embed(src[index_x, index_y].unsqueeze(-1))), -1))), dim=-1)
                d_output = self.model.out(d_outputs.last_hidden_state)
            
            ############ 根据概率融合后的词表输出解码 ################
            # 将S和C概率设置为0
            # d_output[:,:,21128] = 0
            # d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1).expand(a_output.size(0), d_output.size(1), 1), 0)
            # # 概率归一化
            # d_output = F.softmax(d_output, dim=-1)
            # # 将Gen的概率乘上decoder的输出
            # d_output *= a_output[:,:,2].unsqueeze(-1).expand(a_output.size(0), d_output.size(1), 1)
            # # 将Skip的概率赋值给decoder的输出在[BLK]位置的概率
            # d_output[:,:,21128] = a_output[:, :, 0]
            # # 将Copy的概率赋值给decoder的输出在x_hat(i)位置的概率
            # d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1).expand(a_output.size(0), d_output.size(1), 1), a_output[:, :, 1].unsqueeze(-1).expand(a_output.size(0), d_output.size(1), 1))

            # d_output_argmax = torch.argmax(d_output, dim=-1)[:, -1] # (batch, 1)
            # # 输出结果为Copy
            # tag_C = d_output_argmax.cpu() == src[index_x, index_y].cpu()
            # # 输出结果为Skip
            # tag_S = d_output_argmax.cpu() == torch.tensor([21128]*src.size(0))
            # tag_CS = tag_C | tag_S
            # x_hat_delta = torch.zeros(src.size(0), dtype=torch.long).masked_fill(tag_CS, 1)
            # index_y += x_hat_delta
            ############ 根据概率融合后的词表输出解码 ################

            ############ 根据Action的输出解码 ################
            # 将S和C概率设置为0
            d_output[:,:,21128] = 0
            d_output = d_output.scatter(2, src[index_x, index_y].unsqueeze(-1).unsqueeze(-1).expand(a_output.size(0), d_output.size(1), 1), 0)
            # 概率归一化
            d_output = F.softmax(d_output, dim=-1)
            # 推理时根据3个动作的概率
            a_output_argmax = torch.argmax(a_output, dim=-1).squeeze(-1) # (batch,)
            # 输出结果为Copy
            tag_C = a_output_argmax.cpu() == torch.ones(src.size(0), dtype=torch.long)
            # 输出结果为Skip
            tag_S = a_output_argmax.cpu() == torch.zeros(src.size(0), dtype=torch.long)
            tag_CS = tag_C | tag_S
            x_hat_delta = torch.zeros(src.size(0), dtype=torch.long).masked_fill(tag_CS, 1)

        
            outputs, log_scores = self.k_best_outputs(outputs, d_output, log_scores, i, self.setting.beam_k)

            # 给标记为Skip的位置赋上[BLK]的输出
            new_tag_S = torch.zeros_like(outputs, dtype=torch.bool)
            new_tag_S[:, :, i] = tag_S.unsqueeze(-1)
            outputs = outputs.masked_fill(new_tag_S, 21128)
            
            # 给标记为Copy的位置赋上x
            for j in range(src.size(0)):
                if tag_C[j] == True:
                    outputs[j][0][i] = src[index_x, index_y][j]

            # 最后更新index_y
            index_y += x_hat_delta
            for p, item in enumerate(index_y):
                if item >= src.size(1):
                    index_y[p] = src.size(1)-1
            ############ 根据Action的输出解码 ################
            
            ones = (outputs==self.sep_id).nonzero() # 获取输出为102的索引矩阵
            sentence_lengths = torch.zeros([outputs.shape[0], outputs.shape[1]], dtype=torch.long).cuda()
            for vec in ones:
                x = vec[0]
                y = vec[1]
                if sentence_lengths[x][y]==0: # First end symbol has not been found yet
                    sentence_lengths[x][y] = vec[-1] # Position of first end symbol

            num_finished_sentences = [len([s for s in sentence_lengths[_] if s > 0]) for _ in range(sentence_lengths.shape[0])]

            if any(num_finished_sentence == self.setting.beam_k for num_finished_sentence in num_finished_sentences):
                alpha = 0.7
                for index in range(len(num_finished_sentences)):
                    if num_finished_sentences[index]==self.setting.beam_k and batch_outputs[index]=='':
                        div = 1/(sentence_lengths[index].type_as(log_scores[index])**alpha)
                        _, ind = torch.max(log_scores[index] * div, -1)
                        ind = ind.item()
                        length = (outputs[index][ind]==self.sep_id).nonzero()[0].item()
                        batch_outputs[index] = self.tokenizer.decode([data.item() for data in outputs[index][ind][1:length]])
                if all(num_finished_sentence == self.setting.beam_k for num_finished_sentence in num_finished_sentences):
                    break
        
        for index in range(len(num_finished_sentences)):
            if num_finished_sentences[index] != self.setting.beam_k and batch_outputs[index]=='':
                try:
                    length = (outputs[index][0]==self.sep_id).nonzero()[0].item()
                    batch_outputs[index] = self.tokenizer.decode([data.item() for data in outputs[index][0][1:length]])
                except:
                    batch_outputs[index] = ' '

        return batch_outputs