from typing import Dict, List
import itertools

import numpy as np
import json
from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("baidu-indexer")
class SingleIdTokenIndexer(TokenIndexer[int]):
    def __init__(self,
                 namespace: str = 'tokens',
                 lowercase_tokens: bool = False
                 ) -> None:
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self.w2i, self.emb_val = self.load_word_emb()

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        #donothing
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in tokens:
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            indices.append(self.w2i[text])

        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    def load_word_emb(self):
        with open('baidu/word2idx.json', encoding='UTF-8') as inf:
            w2i = json.load(inf)
        with open('baidu/usedwordemb.npy', "rb") as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
