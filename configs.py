from collections import namedtuple

evaluate_gaussian = False
evaluate_random = False
evaluate_DE = False
FSBO = namedtuple("FSBO", "pretrain evaluate")
FSBO.pretrain = False
FSBO.evaluate = False

# List wise ranking loss
RankingLosses = namedtuple("RankingLosses", "pretrain evaluate")
RankingLosses.pretrain = False
RankingLosses.evaluate = False

# Pair wise ranking loss
PairWiseRankingLoss = namedtuple("PairWiseRankingLoss", "pretrain evaluate")
PairWiseRankingLoss.pretrain = False
PairWiseRankingLoss.evaluate = False

plot_ranks = False
