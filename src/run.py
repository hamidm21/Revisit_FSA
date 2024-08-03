import sys
import os
import logging
from argparse import ArgumentParser
from experiment import REGISTERED_EXPERIMENTS
# remember to add the experiment to the registered experiments before calling its id
EXPERIMENTS = {experiment().id: experiment for experiment in REGISTERED_EXPERIMENTS}
# IMPORTANT! change the path to the absolute root path of the project in your system
root_dir = os.path.abspath("$HOME/src/Financial_NLP")
# Add this directory to sys.path
sys.path.append(root_dir)

# Create and configure logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def runner(args):
    id = args.id
    if id in EXPERIMENTS:
        # create logger
        logger = logging.getLogger('experiments logger')
        logger.info(f"experiment {id} found...")
        exp = EXPERIMENTS.get(id)(logger=logger)
        exp.run()
    else:
        raise ValueError("no experiment with the given id exist")

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Experiment Runner",
        description="running the selected experiment",
        epilog="bye!"
    )
    parser.add_argument("id", type=int, help="id of the intended experiment")  # Use the custom help message
    args = parser.parse_args()

    runner(args)
