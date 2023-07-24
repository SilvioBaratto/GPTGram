import argparse
import torch
from GPTGram.train import GramTrainer
from GPTGram.base import DDPContext
from GPTGram.config import Config as cfg
from GPTGram.argparse import arg_parser

def main(args):
    with DDPContext(cfg.ddp.ddp) as ddp_context:
        trainer = GramTrainer(**vars(args))
        trainer.train()

if __name__ == '__main__':
    args = arg_parser()
    main(args)
