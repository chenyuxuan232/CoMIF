import argparse
import torch
from utils.utils import seed_torch, draw_datalogger, DataLogger
from tools.train import train_model
from tools.valid import valid_model
from tools.predict import do_predict
from tools.eval import do_evaluation
from configs.configs import TrainConfig, ValidConfig, PredictConfig, EvalConfig

TRAIN = True
VALID = True
PREDICT = False
EVAL = True


def main(args):
    seed_torch(0)

    factor = args.factor
    dataset = args.dataset_name
    device = args.device

    # factor = "PE"
    # # factor = None
    # dataset = "PC"  # ['SPC', 'PC']
    # device = "cuda:7"
    print("factor:", factor, "dataset:", dataset)

    if TRAIN:
        loadFilename = None
        epoch = 5
        train_config = TrainConfig(
            factor=factor,
            dataset=dataset,
            epoch=epoch,
            loadFilename=loadFilename,
            device=device,
        )
        print(train_config.model_name)
        train_model(train_config)
        torch.cuda.empty_cache()

    if VALID:
        model_name = "GPT2" if factor is None else f"{factor}-GPT2"
        epoch_idx = 5
        valid_config = ValidConfig(
            model_name=model_name,
            dataset=dataset,
            epoch_idx=epoch_idx,
            device=device,
        )
        valid_model(valid_config)
        torch.cuda.empty_cache()

    if PREDICT:
        model_name = "GPT2" if factor is None else f"{factor}-GPT2"
        # epoch_idx = 5
        for epoch_idx in range(1, 6):
            predict_config = PredictConfig(
                model_name=model_name,
                dataset=dataset,
                epoch_idx=epoch_idx,
                device=device,
            )
            do_predict(predict_config)
            torch.cuda.empty_cache()

    if EVAL:
        model_name = "GPT2" if factor is None else f"{factor}-GPT2"
        # epoch_idx = 5
        for epoch_idx in range(1, 6):
            eval_config = EvalConfig(
                model_name=model_name,
                dataset=dataset,
                epoch_idx=epoch_idx,
                device=device,
            )
            do_evaluation(eval_config)
            torch.cuda.empty_cache()


# def show_loss():
#     loadFilename = "outputs/checkpoints/PTE-GPT2/PC-5e.ckpt"
#     model_name = loadFilename.split("/")[-2]
#     ckpt = torch.load(loadFilename, map_location="cpu")
#     datalogger = DataLogger()
#     datalogger.__dict__ = ckpt["loss_logger"]
#     draw_datalogger(datalogger, model_name, "reconstruction_loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--factor", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
    # show_loss()
