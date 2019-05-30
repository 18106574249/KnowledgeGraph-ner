class DefaultConfig(object):
    data_path='./data/train_data.txt'
    emb_path='./data/facebook_wiki/zh.300'

    lr = 2e-5
    batch_size = 64
    num_epochs = 20
    output_size = 2
    hidden_size = 128
    dropout = 0
    embedding_length = 0
    embeddings = None
    vocab_size = 0
    # ealystopping的等待和阈值
    patience = 10
    min_delta = 1e-3
    def parse(self, kwargs):

        '''
        根据字典kwargs更新config参数
        '''

        # hasattr 判断object是否有这个属性
        for k, v in kwargs.items():
            if not hasattr(self,k):
                print('Warning:opt has not attribute %s' %k)
            setattr(self, k, v)
