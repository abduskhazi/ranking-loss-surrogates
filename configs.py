from collections import namedtuple

evaluate_gaussian = False
evaluate_random = False
evaluate_DE = False
FSBO = namedtuple("FSBO", "pretrain evaluate")
FSBO.pretrain = False
FSBO.evaluate = False

RankingLosses = namedtuple("RankingLosses", "pretrain evaluate")
RankingLosses.pretrain = False
RankingLosses.evaluate = True

plot_ranks = True
