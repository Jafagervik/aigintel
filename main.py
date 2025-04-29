import logging

from tinygrad import Device

from aigintel import utils
from aigintel.inference import run_multiple
from aigintel.models.example_model import TinyNet
from aigintel.train import train
from aigintel.utils import load_config, seed_all


def main():
    args = utils.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = load_config("config.yaml")

    seed_all(config["seed"])

    logging.info("Training..." if args.train else "Inference...")
    logging.debug(f"Default device is {Device.DEFAULT}")

    # TODO: Replace with your net
    model = TinyNet()
    # model = LinearNet()

    load_config("config.yaml")

    if args.train:
        train(model, config, args)
    else:
        run_multiple(model, config, args)
        # run(model, config, args)


if __name__ == "__main__":
    main()
