import argparse
import torch
from utils.utils import seed_torch, draw_datalogger, DataLogger
from tools.train import train_model
from tools.valid import valid_model
from tools.predict import do_predict
from tools.eval import do_evaluation
from configs.configs import TrainConfig, ValidConfig, PredictConfig, EvalConfig

PREDICT = True
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
    print("消融实验")
    print(f"no_P: {args.no_P}\tno_T: {args.no_T}\tno_E: {args.no_E}\t")

    if PREDICT:
        model_name = "GPT2" if factor is None else f"{factor}-GPT2"
        # epoch_idx = 5
        for epoch_idx in range(5, 6):
            predict_config = PredictConfig(
                model_name=model_name,
                dataset=dataset,
                epoch_idx=epoch_idx,
                device=device,
                no_P=args.no_P,
                no_T=args.no_T,
                no_E=args.no_E,
            )
            do_predict(predict_config)
            torch.cuda.empty_cache()

    if EVAL:
        model_name = "GPT2" if factor is None else f"{factor}-GPT2"
        # epoch_idx = 5
        for epoch_idx in range(5, 6):
            eval_config = EvalConfig(
                model_name=model_name,
                dataset=dataset,
                epoch_idx=epoch_idx,
                device=device,
                no_P=args.no_P,
                no_T=args.no_T,
                no_E=args.no_E,
            )
            do_evaluation(eval_config)
            torch.cuda.empty_cache()


def show_loss():
    loadFilename = "outputs/checkpoints/PTE-GPT2/PC-5e.ckpt"
    model_name = loadFilename.split("/")[-2]
    ckpt = torch.load(loadFilename, map_location="cpu")
    datalogger = DataLogger()
    datalogger.__dict__ = ckpt["loss_logger"]
    draw_datalogger(datalogger, model_name, "reconstruction_loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--factor", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--no_P", action="store_true", default=False)
    parser.add_argument("--no_T", action="store_true", default=False)
    parser.add_argument("--no_E", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    # show_loss()
