import argparse
from scipy.stats import pearsonr, spearmanr

def main(args):
    h_scores = list(map(float, open(args.human).read().rstrip().split(',')))
    s_scores = list(map(float, open(args.system).read().rstrip()[:-1].split(',')))
    print('Pearson ', round(pearsonr(h_scores, s_scores)[0], 4))
    print('Spearman', round(spearmanr(h_scores, s_scores)[0], 4))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', required=True)
    parser.add_argument('--system', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)