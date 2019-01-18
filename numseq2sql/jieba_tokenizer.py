from typing import List

from allennlp.data.tokenizers.word_filter import PassThroughWordFilter, WordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.word_stemmer import PassThroughWordStemmer, WordStemmer
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

from numseq2sql.jieba_splitter import JiebaSplitter


'''
重写tokenizer
'''
@Tokenizer.register("jieba_tokenizer")
class JiebaTokenizer(Tokenizer):

    def __init__(self,
                 word_splitter: WordSplitter = None,
                 word_filter: WordFilter = PassThroughWordFilter(),
                 word_stemmer: WordStemmer = PassThroughWordStemmer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._word_splitter = word_splitter or JiebaSplitter()
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        words = self._word_splitter.split_words(text)
        return self._filter_and_stem(words)

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        batched_words = self._word_splitter.batch_split_words(texts)
        return [self._filter_and_stem(words) for words in batched_words]

    def _filter_and_stem(self, words: List[Token]) -> List[Token]:
        filtered_words = self._word_filter.filter_words(words)
        stemmed_words = [self._word_stemmer.stem_word(word) for word in filtered_words]
        for start_token in self._start_tokens:
            stemmed_words.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            stemmed_words.append(Token(end_token, -1))
        return stemmed_words