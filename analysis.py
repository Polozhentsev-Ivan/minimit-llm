import json, argparse, statistics
from metrics.meteor import MeteorMetric

def read_columns(path, gold_key, pred_key):
    gold, pred = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            gold.append(obj[gold_key])
            pred.append(obj[pred_key])
    return gold, pred

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--gold_key', default='gold')
    p.add_argument('--pred_key', default='generated')
    args = p.parse_args()

    refs, hyps = read_columns(args.data, args.gold_key, args.pred_key)

    metric = MeteorMetric()
    scores = metric.compute_scores(hyps, refs)

    report = {
        'samples': len(scores),
        'mean_meteor': statistics.mean(scores),
        'median_meteor': statistics.median(scores),
        'best': max(scores),
        'worst': min(scores)
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()