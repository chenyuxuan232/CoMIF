# 以下参数在其他代码中会被直接调用
emotion_label_id_map = {
    "admiration": 0,
    "amusement": 1,
    "anger": 2,
    "annoyance": 3,
    "approval": 4,
    "caring": 5,
    "confusion": 6,
    "curiosity": 7,
    "desire": 8,
    "disappointment": 9,
    "disapproval": 10,
    "disgust": 11,
    "embarrassment": 12,
    "excitement": 13,
    "fear": 14,
    "gratitude": 15,
    "grief": 16,
    "joy": 17,
    "love": 18,
    "nervousness": 19,
    "optimism": 20,
    "pride": 21,
    "realization": 22,
    "relief": 23,
    "remorse": 24,
    "sadness": 25,
    "surprise": 26,
    "neutral": 27,
}
emotion_threshold_dict = {
    "admiration": 0.40,
    "amusement": 0.35,
    "anger": 0.15,
    "annoyance": 0.20,
    "approval": 0.15,
    "caring": 0.05,
    "confusion": 0.10,
    "curiosity": 0.15,
    "desire": 0.05,
    "disappointment": 0.10,
    "disapproval": 0.15,
    "disgust": 0.05,
    "embarrassment": 0.05,
    "excitement": 0.10,
    "fear": 0.10,
    "gratitude": 0.65,
    "grief": 0.05,
    "joy": 0.15,
    "love": 0.15,
    "nervousness": 0.05,
    "optimism": 0.10,
    "pride": 0.05,
    "realization": 0.05,
    "relief": 0.05,
    "remorse": 0.20,
    "sadness": 0.10,
    "surprise": 0.10,
    "neutral": 0.25,
}
n_emotion_class = len(emotion_label_id_map)


# 通用参数
EMOTION_N_CLASS = n_emotion_class
ENCODER_MODEL_PATH = "model_hubs/roberta-base"
DECODER_MODEL_PATH = "model_hubs/gpt2"
OUTPUT_DIR = "outputs_new"
# 检查点保存路径
CKPT_SAVE_DIR = f"{OUTPUT_DIR}/checkpoints"
# 数据集保存路径
SPC_DATASET_DIR = "data/Synthetic-Persona-Chat-splited"
PC_DATASET_DIR = "data/Persona-Chat_splited"
# 验证结果保存路径
VALID_OUTPUT_DIR = f"{OUTPUT_DIR}/valid"
# 预测结果保存路径
PREDICT_OUTPUT_DIR = f"{OUTPUT_DIR}/predicts"
# 评估结果保存路径
EVAL_OUTPUT_DIR = f"{OUTPUT_DIR}/eval"
# 消融实验结果保存路径
ABLATION_OUTPUT_DIR = f"{OUTPUT_DIR}/Ablation Study"
# 损失权重
REC = 10
EM_HIS = 1
KW_HIS = 1
EM_G = 0
KW_G = 0


class TrainConfig:
    def __init__(
        self,
        *,
        factor=None,
        dataset="SPC",
        epoch=5,
        batch_size=12,
        learning_rate=1e-4,
        loadFilename=None,
        device=None,
    ):
        if factor is None:
            factor = ""
        self.factor = ""
        if "S" in factor:
            self.factor += "S"
        if "P" in factor:
            self.factor += "P"
        if "T" in factor:
            self.factor += "T"
        if "E" in factor:
            self.factor += "E"
        self.dataset = dataset

        self.epoches = epoch
        self.batch_size = batch_size

        self.ckpt_save_dir = CKPT_SAVE_DIR
        self.model_name = f"{factor}-GPT2" if factor != "" else "GPT2"
        self.loadFilename = loadFilename  # load train_info from checkpoint

        self.learning_rate = learning_rate
        self.clip = 50

        self.device = device if device is not None else "cpu"


class ValidConfig:
    def __init__(self, *, model_name: str, dataset: str, epoch_idx: int, device=None):
        self.model_name = model_name
        self.dataset = dataset
        self.epoch_idx = epoch_idx

        self.batch_size = 1

        self.ckpt_dir = CKPT_SAVE_DIR
        self.valid_output_dir = VALID_OUTPUT_DIR

        self.device = device if device is not None else "cpu"

        # 根据model_name获取
        factor = self.model_name.split("GPT2")[0]
        self.factor = ""
        if "S" in factor:
            self.factor += "S"
        if "P" in factor:
            self.factor += "P"
        if "T" in factor:
            self.factor += "T"
        if "E" in factor:
            self.factor += "E"


class PredictConfig:
    def __init__(
        self,
        *,
        model_name: str,
        dataset: str,
        epoch_idx: int,
        max_generate_length: int = 32,
        device=None,
        no_P=False,
        no_T=False,
        no_E=False,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.epoch_idx = epoch_idx

        self.batch_size = 1

        self.ckpt_dir = CKPT_SAVE_DIR
        self.predict_output_dir = PREDICT_OUTPUT_DIR

        self.device = device if device is not None else "cpu"

        # 根据model_name获取
        factor = self.model_name.split("GPT2")[0]
        self.factor = ""
        if "S" in factor:
            self.factor += "S"
        if "P" in factor:
            self.factor += "P"
        if "T" in factor:
            self.factor += "T"
        if "E" in factor:
            self.factor += "E"

        self.max_generate_length = max_generate_length

        self.ablation = {"no_P": no_P, "no_T": no_T, "no_E": no_E}
        if no_P or no_T or no_E:
            tmp_dir = ""
            if no_P:
                tmp_dir += "P"
            if no_T:
                tmp_dir += "T"
            if no_E:
                tmp_dir += "E"
            self.predict_output_dir = ABLATION_OUTPUT_DIR + f"/{tmp_dir}"


class EvalConfig:
    def __init__(
        self,
        *,
        model_name: str,
        dataset: str,
        epoch_idx: int,
        batch_size: int = 8,
        device=None,
        no_P=False,
        no_T=False,
        no_E=False,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.epoch_idx = epoch_idx

        self.batch_size = batch_size

        self.ckpt_dir = CKPT_SAVE_DIR
        self.predict_dir = PREDICT_OUTPUT_DIR
        self.eval_output_dir = EVAL_OUTPUT_DIR

        self.device = device if device is not None else "cpu"

        # 根据model_name获取
        factor = self.model_name.split("GPT2")[0]
        self.factor = ""
        if "S" in factor:
            self.factor += "S"
        if "P" in factor:
            self.factor += "P"
        if "T" in factor:
            self.factor += "T"
        if "E" in factor:
            self.factor += "E"

        self.ablation = {"no_P": no_P, "no_T": no_T, "no_E": no_E}
        if no_P or no_T or no_E:
            tmp_dir = ""
            if no_P:
                tmp_dir += "P"
            if no_T:
                tmp_dir += "T"
            if no_E:
                tmp_dir += "E"
            self.predict_dir = ABLATION_OUTPUT_DIR + f"/{tmp_dir}"
            self.eval_output_dir = ABLATION_OUTPUT_DIR + f"/{tmp_dir}"


###################################################################
# 以下两个 config 类分别用于 tools.utils 中获取模型和数据集, 均无需修改
###################################################################
class DataConfig:
    def __init__(self, *, dataset, split, batch_size, shuffle, all_raw_datas=None):
        assert dataset in ["SPC", "PC"], "数据集只能是SPC或PC"
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.encoder_model_path = ENCODER_MODEL_PATH
        self.decoder_model_path = DECODER_MODEL_PATH

        self.dataset_dir = SPC_DATASET_DIR if dataset == "SPC" else PC_DATASET_DIR

        self.utterance_truncate: int = 8
        self.context_utterance_length: int = 64
        self.input_utterance_length: int = 450
        self.persona_length: int = 32

        self.all_raw_datas = all_raw_datas


class ModelConfig:
    def __init__(self, *, factor):
        """
        factor 可以是 "SPTE" 的组合或 'None'
        factor 是 str 时，其中只要包含 'S','P','T','E' 等字符即可
        如 "SP78T& ^*@" 会被识别成 "SPT"
        """
        self.encoder_model_path = ENCODER_MODEL_PATH
        self.decoder_model_path = DECODER_MODEL_PATH

        self.rnn_n_layers = 2
        self.emotion_n_class = EMOTION_N_CLASS

        self.factor = factor
        self.dropout = 0.1

        self.freeze_enc = True
        self.freeze_dec = False
