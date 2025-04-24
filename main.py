from typing import List

from tinygrad import Device

from aigintel import utils
from aigintel.inference import run
from aigintel.models.model import LinearNet

from aigintel.train import train
from aigintel.utils import load_config, seed_all

def main():
    args = utils.parse_args()
    config = load_config("config.yaml")

    seed_all(config["seed"])

    print("Training..." if args.train else "Inference...")

    if args.debug:
        print(f"Default device is {Device.DEFAULT}")

    #model = LinearNet(config["layers"])
    model = LinearNet()

    if args.load:

        load_config("config.yaml")

    if args.train:
        # Training
        train(model, config, args)
    else:
        # Inference
        run(model, None, config, args)


if __name__ == '__main__':
    main()
