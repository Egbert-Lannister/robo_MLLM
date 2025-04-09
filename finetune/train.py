import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.schemas import Args
# from i2va_trainer import I2VATrainer
from finetune.trainers.trainer import CogVideoXImageToVideoActionTrainer


def main():
    args = Args.parse_args()
    trainer = CogVideoXImageToVideoActionTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
