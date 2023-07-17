from GPTGram.sample import GramSampler
from GPTGram.base import DDPContext
from GPTGram.config import Config as cfg
from GPTGram.argparse import arg_parser

def main(args):
    with DDPContext(cfg.ddp.ddp) as ddp_context:
        sampler = GramSampler(**vars(args))
        sampler.generate(temperature=cfg.sampling.temperature,
                         top_k=cfg.sampling.top_k
                         )

if __name__ == '__main__':
    args = arg_parser()
    main(args)
