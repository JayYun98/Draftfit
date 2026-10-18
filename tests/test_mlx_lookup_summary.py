import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.summarize_mlx_lookup import summarize


class LookupSummaryTest(unittest.TestCase):
    def test_partial_run_is_not_a_completed_result(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'run.jsonl'
            p.write_text('{"kind":"provenance"}\n')
            with self.assertRaisesRegex(ValueError, 'did not finish'):
                summarize(p)

    def test_acceptance_is_token_weighted_and_sources_separate(self):
        row = dict(kind='measurement', group='general', domain='code', arm='hybrid',
                   id='one', trial=0, parity=True, tok_s=10, decode_tok_s=20, seconds=2,
                   result={'num_tokens':20}, peak_memory_bytes=100,
                   rounds=[{'source':'lookup','drafted':1,'accepted':1},
                           {'source':'lookup','drafted':9,'accepted':0},
                           {'source':'drafter','drafted':2,'accepted':2}])
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'run.jsonl'
            p.write_text(json.dumps(row)+'\n'+json.dumps({'kind':'complete'})+'\n')
            s = summarize(p)
        self.assertEqual(s['measurements'], 1)
        self.assertEqual(s['parity_failures'], 0)
        for group in s['groups']:
            self.assertEqual(group['sources']['lookup']['acceptance'], .1)
            self.assertEqual(group['sources']['drafter']['acceptance'], 1)
            self.assertIsNone(group['sources']['plain']['acceptance'])

if __name__ == '__main__': unittest.main()
