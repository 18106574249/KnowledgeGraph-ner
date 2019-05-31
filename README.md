# KnowledgeGraph-ner
Named entity recognition in pytorch

Github地址：https://github.com/feifeife/KnowledgeGraph-ner

本项目使用
* python3.7
* pytorch 1.1.0
实现了命名实体识别的Bilstm+crf layer的模型

## 文件架构
- `/data`下存储原始数据集和输入数据集，最终输入数据文件为train.txt和test.txt文件
- `/data/facebook_wiki`下存放预训练好的词向量，embedding size为300
- `config.py`存储参数设置
-  `bilstm_crf.py`和`crf.py`定义模型结构
-  `train.py`训练模型和test
-  数据处理及utils放在`data_process.py`
-  `word_embedding.py`用来读取预训练好的词向量

## 数据来源
* 原始数据

原始数据来源于MSRA的开源数据,位于/data下

|    | sentence | PER | LOC | ORG |
| :----: | :---: | :---: | :---: | :---: |
| train  | 46364 | 17615 | 36517 | 20571 |
| test   | 4365  | 1973  | 2877  | 1331  |

对于每一个中文字符，都对应一个标记，{O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}

* 数据格式如下：
```
因/o 有关/o 日/ns 寇/o 在/o 京/ns 掠夺/o 文物/o 详情/o ，/o 藏/o 界/o 较为/o 重视/o ，/o 也是/o 我们/o 收藏/o 北京/ns 史料/o 中/o 的/o 要件/o 之一/o 。/o 
```

* 最终输入数据格式：

```
因	O
有	O
关	O
日	B-LOC
寇	O
在	O
京	B-LOC
掠	O
```
* 词向量

使用预训练好的中文词向量，训练预料为Wikipedia，位于/data/facebook_wiki
## 模型
实现了经典的bilstm+crf模型

![Network](./pic1.png)


## 训练
训练好的模型存在/checkpoints 下

模型在测试集的最终表现如下：

| Precision     | Recall     | F1     | 
| :---: | :---: | :---: |
| 0.9122 | 0.9035 | 0.9126 |

