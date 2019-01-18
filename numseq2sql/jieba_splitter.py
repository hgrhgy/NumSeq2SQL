import jieba
from allennlp.data.tokenizers.word_splitter import WordSplitter
import jieba.posseg as pseg
from overrides import overrides
from typing import List
from allennlp.data.tokenizers.token import Token


def _remove_spaces(tokens: List[Token]) -> List[Token]:
    return [token for token in tokens if not token.text.isspace()]

'''
重写Splitter
'''
@WordSplitter.register('jieba_splitter')
class JiebaSplitter(WordSplitter):
    def __init__(self, cut_all = False, dict = None):
        super(JiebaSplitter, self).__init__()
        self.cut_all = cut_all
        self.dict = dict
        if self.dict is not None:
            jieba.load_userdict(dict)

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [_remove_spaces(self.split_words(s)) for s in sentences]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        ## jieba pair (text, pos)
        tokens = [Token(text=p.word, pos=p.flag) for p in list(pseg.cut(sentence))]
        return _remove_spaces(tokens)
