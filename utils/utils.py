import random
import os
import numpy as np
import torch
from tqdm import tqdm


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_ignore_index(ids, attn_mask=None, *, token_type_ids=None, ignore_index=-100):
    assert (
        attn_mask is not None or token_type_ids is not None
    ), "'attn_mask' and 'token_type_ids' cannot be 'None' at the same time"

    tmp_ids = ids.clone().detach()
    if token_type_ids is None:
        tmp_ids[attn_mask == 0] = ignore_index
    else:
        tmp_ids[token_type_ids == 0] = ignore_index
    return tmp_ids


class DataLogger:
    def __init__(self, num: int = 0):
        """
        num = 0: .avg()计算全部数据的均值
        """
        super(DataLogger, self).__init__()
        self.datas = []
        self.num = num

    def add(self, data):
        self.datas.append(data)

    def avg(self, key=None, num=None):
        num = self.num if num is None else num

        datas = self.datas[-num:]
        if len(datas) == 0:
            return 0

        # 如果当前Logger保存的是一个dict，根据key取数据
        if type(datas[0]) is dict:
            assert key is not None, "请选择需要输出的数据"
            datas = [data[key] for data in datas]

        return sum(datas) / len(datas)

    def avg_all(self, key=None):
        if len(self.datas) == 0:
            return 0

        datas = self.datas
        # 如果当前Logger保存的是一个dict，根据key取数据
        if type(datas[0]) is dict:
            assert key is not None, "请选择需要输出的数据"
            datas = [data[key] for data in datas]
        return sum(datas) / len(datas)


def pack_batch(batches, real_num):
    """
    Params:
        batches: (bsz, max_num, seq_len)
        real_num: (bsz)
    Return:
        packed_batches: (bsz * max_num - pad_num, seq_len)
    """
    # packed_batches 中每个元素的 shape 为 (num, seq_len)
    packed_batches = []
    for batch, num in zip(batches, real_num):
        packed_batches.append(batch[:num])

    packed_batches = torch.cat(packed_batches, dim=0)
    return packed_batches


def pad_batch(packed_batches, real_num, max_num: int, pad_id):
    """
    Params:
        packed_batches: (bsz * max_num - pad_num, seq_len)
        real_num: (bsz)
        max_num: int
    Return:
        packed_batches: (bsz, max_num, seq_len)
    """
    input_dims = len(packed_batches.shape)
    # padded_batches 中每个元素的 shape 为 (1, max_num, seq_len)
    padded_batches = []
    start = 0
    for num in real_num:
        num_padding = max_num - num
        pad_tuple = tuple(([0] * (2 * input_dims - 1)) + [num_padding])

        padded_batch = packed_batches[start : start + num]
        padded_batch = torch.nn.functional.pad(padded_batch, pad_tuple, value=pad_id)
        padded_batch = padded_batch.unsqueeze(0)

        padded_batches.append(padded_batch)
        start = start + num

    padded_batches = torch.cat(padded_batches, dim=0)
    return padded_batches


def ignore_warning():
    import warnings

    warnings.filterwarnings("ignore")


import numpy as np
from torch.utils.tensorboard import SummaryWriter


def draw_datalogger(datalogger: DataLogger, name:str, key=None):
    writer = SummaryWriter("logs/loss")

    if type(datalogger.datas[0]) is dict:
        assert key is not None, "没有选择需要绘制的key"
        losses = [data[key] for data in datalogger.datas]
        name = name + "_" + key
    else:
        assert key is None, "该datalogger的数据不是dict"
        losses = datalogger.datas
    # x = range(len(y))
    step = 0
    for i in tqdm(range(len(losses))):
        i = i + 1
        tmp_loss = losses[i - 50 if i - 50 > 0 else 0 : i]
        loss = sum(tmp_loss) / len(tmp_loss)
        writer.add_scalar(name, loss, step)
        step += 1
    writer.close()
