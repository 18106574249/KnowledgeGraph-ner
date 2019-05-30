import bcolz
import numpy as np
def load_embeddings(folder_path):
    """从 bcolz 加载 词/字 向量

    Args:
        - folder_path (str): 解压后的 bcolz rootdir（如 zh.64），
                             里面包含 2 个子目录 embeddings 和 words，
                             分别存储 嵌入向量 和 词（字）典

    Returns:
        - words (bcolz.carray): 词（字）典列表（bcolz carray  具有和 numpy array 类似的接口）
        - embeddings (bcolz.carray): 嵌入矩阵，每 1 行为 1 个 词向量/字向量，
                                     其行号即为该 词（字） 在 words 中的索引编号
    """
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words'%folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%folder_path, mode='r')
    words = np.asarray(words).tolist()
    embeddings = np.asarray(embeddings).tolist()

    return words,embeddings
