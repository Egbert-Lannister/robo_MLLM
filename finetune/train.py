import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.schemas import Args
# from finetune.trainers.robo_closed_loop_trainer import RoboClosedLoopTrainer
# from finetune.trainers.i2va_trainer import I2VATrainer
from finetune.trainers.robo_multimodel_trainer import RoboMultimodelTrainer

def main():
    args = Args.parse_args()
    trainer = RoboMultimodelTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
