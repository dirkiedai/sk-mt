# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer


@register_scorer("edistance")
class EDistanceScorer(BaseScorer):
    def __init__(self, args):
        super().__init__(args)
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=self.args.ed_tokenizer,
            lowercase=self.args.ed_lowercase,
            punctuation_removal=self.args.ed_remove_punct,
            character_tokenization=self.args.ed_char_level,
        )

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--ed-tokenizer', type=str, default='none',
                            choices=EvaluationTokenizer.ALL_TOKENIZER_TYPES,
                            help='sacreBLEU tokenizer to use for evaluation')
        parser.add_argument('--ed-remove-punct', action='store_true',
                            help='remove punctuation')
        parser.add_argument('--ed-char-level', action='store_true',
                            help='evaluate at character level')
        parser.add_argument('--ed-lowercase', action='store_true',
                            help='lowercasing')
        # fmt: on

    def reset(self):
        self.distance = 0
        self.examples = 0

    def add_string(self, ref, pred):
        ref_items = self.tokenizer.tokenize(ref).split()
        pred_items = self.tokenizer.tokenize(pred).split()
        self.distance += self.ed.eval(ref_items, pred_items)
        self.examples += 1

    def result_string(self):
        return f"Edit Distance: {self.score():.2f}"

    def score(self):
        return float(self.distance) / self.examples if self.examples > 0 else 0 
