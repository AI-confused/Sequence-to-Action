# Sequence-to-Action
## Description
#### 1. 该代码库是基于腾讯在AAAI 2022发表的一篇关于中文语法纠错的论文来复现的，有些小细节可能会与论文有出入，笔者已经尽可能还原论文作者的思路
#### 2. 论文为：[Sequence-to-Action: Grammatical Error Correction with Action Guided Sequence Generation](https://arxiv.org/abs/2205.10884)
#### 3. 笔者在参加CCL 2022的中文语病诊断评测任务中，通过复现该论文的方法来提升单模指标，可能由于数据量或者训练方式的问题，使用原生的Transformer模型几乎看不到效果，因此将该部分替换成Bart，才有明显的效果，且指标高于官方提供的基线方法-Gector
___
## 工程结构
该工程代码库主要基于[easy-task训练框架](https://github.com/AI-confused/easy_task)，可实现灵活地搭建训练任务
### config
    - grammar_s2a_bart.yml 任务的训练配置文件
    - grammar_s2a_predict.yml 任务的预测配置文件
### src
    - correct_result.py 任务的预测结果评估文件
    - model.py 存放模型的文件
    - seq2action_bart_task.py 任务的训练和预测代码
    - run_task_s2a.py 任务的入口文件
___
## 运行方法
### 训练任务
#### 1. grammar_s2a_bart.yml配置好训练参数
#### 2. 选择该配置文件进行加载 task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_s2a_bart.yml'))
#### 3. python3 src/run_task_s2a.py
### 预测任务
#### 1. grammar_s2a_predict.yml配置好训练参数
#### 2. 选择该配置文件进行加载 task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_s2a_predict.yml'))
#### 3. python3 src/run_task_s2a.py
## 实验结果