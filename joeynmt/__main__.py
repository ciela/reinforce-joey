import argparse

from joeynmt.training import train
from joeynmt.prediction import test
from joeynmt.prediction import translate


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "translate"],
                    help="train a model or test or translate")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    ap.add_argument("--ckpt", type=str,
                    help="checkpoint for prediction")

    ap.add_argument("--alpha", type=float,
                    help="beam alpha")

    ap.add_argument("--output_path", type=str,
                    help="path for saving translation output")

    ap.add_argument("--save_attention", action="store_true",
                    help="save attention visualizations")

    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path, alpha=args.alpha)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, alpha=args.alpha,
             output_path=args.output_path, save_attention=args.save_attention)
    elif args.mode == "translate":
        translate(cfg_file=args.config_path, ckpt=args.ckpt, alpha=args.alpha,
                  output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
