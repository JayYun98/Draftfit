"""Aggregate completed lookup benchmark JSONL without conflating draft sources."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if rows[-1].get('kind') != 'complete':
        raise ValueError('benchmark did not finish')
    measurements = [r for r in rows if r['kind'] == 'measurement']
    buckets = collections.defaultdict(list)
    for r in measurements:
        buckets[(r['group'], r['arm'])].append(r)
        if r['group'] == 'general':
            buckets[('domain:'+r['domain'], r['arm'])].append(r)
    baseline = {(r["id"], r["trial"]): r for r in measurements if r["arm"] == "target"}
    result=[]
    for (group,arm),items in sorted(buckets.items()):
        sources={}
        for source in ['lookup','drafter','plain']:
            rounds=[x for r in items for x in r['rounds'] if x['source']==source]
            proposed=sum(x['drafted'] for x in rounds)
            accepted=sum(x['accepted'] for x in rounds)
            sources[source]=dict(rounds=len(rounds),proposed=proposed,accepted=accepted,
                                 acceptance=accepted/proposed if proposed else None)
        paired = [r["tok_s"] / baseline[(r["id"], r["trial"])]["tok_s"]
                  for r in items if (r["id"], r["trial"]) in baseline]
        result.append(dict(median_paired_speedup=statistics.median(paired) if paired else None, group=group,arm=arm,requests=len(items),cases=len({x['id'] for x in items}),
                           parity=all(r['parity'] for r in items),
                           median_tok_s=statistics.median(r['tok_s'] for r in items),
                           min_tok_s=min(r['tok_s'] for r in items),
                           max_tok_s=max(r['tok_s'] for r in items),
                           median_latency_s=statistics.median(r['seconds'] for r in items),
                           p95_latency_s=sorted(r['seconds'] for r in items)[max(0,math.ceil(.95*len(items))-1)],
                           output_tokens=sum(r['result']['num_tokens'] for r in items),
                           sources=sources))
    return dict(timing='end_to_end_including_prefill; legacy raw decode fields are invalid',raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                measurements=len(measurements),parity_failures=sum(not r['parity'] for r in measurements),
                max_peak_memory_bytes=max(r['peak_memory_bytes'] for r in measurements),groups=result)

if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
