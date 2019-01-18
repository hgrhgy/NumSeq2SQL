import json
import logging
from typing import Dict, List, Tuple, Any

import jieba
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import TokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField

from numseq2sql.jieba_splitter import JiebaSplitter
from numseq2sql.jieba_tokenizer import JiebaTokenizer

logger = logging.getLogger(__name__)

'''
重写reader
'''


@DatasetReader.register("num_reader")
class NumReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or JiebaTokenizer(word_splitter=JiebaSplitter())
        self._token_indexers = token_indexers or {'tokens': PretrainedBertIndexer(
            "./bert/multi_cased_L-12_H-768_A-12/bert-base-multilingual-cased-vocab.txt")}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path, encoding="UTF-8") as dataset_file:
            dataset_json = json.load(dataset_file)
            metadata = dataset_json['table']
            dataset = dataset_json['data']

        ## build dict by metadata
        # for y in metadata['years']:
        #     jieba.add_word(y, tag='YE')
        #     jieba.suggest_freq(y, True)
        # for a in metadata['areas']:
        #     jieba.add_word(a, tag='AR')
        #     jieba.suggest_freq(a, True)

        logger.info("Reading the dataset")

        debug = 0
        col_str = ",".join(metadata["columns"])
        tokenized_columns = self._tokenizer.tokenize(col_str)

        for paragraph_json in dataset:
            paragraph = paragraph_json["passage"]
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            seqandsql = paragraph_json['seqandsql']

            sentences = []
            sqls = []
            column_start_spans = []
            column_end_spans = []
            value_start_spans = []
            value_end_spans = []
            yesno_list = []

            tokenized_sentence = []


            for s in seqandsql:
                sentences.append(s["sentence"])
                sentence_tokens = self._tokenizer.tokenize(s["sentence"])
                tokenized_sentence.append(sentence_tokens)

                sqls.append(s["sql"])
                col_tokens = self._tokenizer.tokenize(s["sql"]['column'])

                col_start_idx, col_end_idx = self.find_passage_token_index(tokenized_paragraph, sentence_tokens,
                                                                           col_tokens)
                #print(''.join([t.text for t in tokenized_paragraph[col_start_idx:col_end_idx]]))
                column_start_spans.append(col_start_idx)
                column_end_spans.append(col_end_idx)

                # value_tokens = self._tokenizer.tokenize(s["sql"]['value'])
                # val_start_idx, val_end_idx = self.find_passage_token_index(tokenized_paragraph, sentence_tokens,
                #                                                            value_tokens)
                # value_start_spans.append(val_start_idx)
                # value_end_spans.append(val_end_idx)
                value_start_spans.append(0)
                value_end_spans.append(0)

                # col_text = s["sql"]['column']
                # beg = col_text.find(s["sentence"])
                yesno_list.append("x")

            instance = self.text_to_instance(sentences,
                                             paragraph,
                                             col_str,
                                             column_start_spans,
                                             column_end_spans,
                                             value_start_spans,
                                             value_end_spans,
                                             sqls,
                                             tokenized_paragraph,
                                             tokenized_columns,
                                             tokenized_sentence,
                                             yesno_list,
                                             metadata)
            yield instance

        # print("debug")
        # print(debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[str],
                         passage: str,
                         columns: str,
                         column_start_spans,
                         column_end_spans,
                         value_start_spans,
                         value_end_spans,
                         sqls,
                         passage_tokens: List[Token] = None,
                         column_tokens: List[Token] = None,
                         sentence_tokens: List[List[Token]] = None,
                         yesno_list: List[int] = None,
                         metadata: Dict[str, Any] = None) -> Instance:

        passage_field = TextField(passage_tokens, self._token_indexers)
        columns_field =TextField(column_tokens, self._token_indexers)
        sentences_field = ListField([TextField(s_tokens, self._token_indexers) for s_tokens in sentence_tokens])

        fields = {'passage': passage_field, 'sentence': sentences_field, 'column': columns_field}
        col_start_list = []
        col_end_list = []
        for s, e in zip(column_start_spans, column_end_spans):
            col_start_list.append(IndexField(s, passage_field))
            col_end_list.append(IndexField(e, passage_field))
        fields['col_start_idx'] = ListField(col_start_list)
        fields['col_end_idx'] = ListField(col_end_list)

        val_start_list = []
        val_end_list = []
        for s, e in zip(value_start_spans, value_end_spans):
            val_start_list.append(IndexField(s, passage_field))
            val_end_list.append(IndexField(e, passage_field))

        fields['val_start_idx'] = ListField(val_start_list)
        fields['val_end_idx'] = ListField(val_end_list)

        metadata['origin_passage'] = passage
        metadata['passage_tokens'] = passage_tokens
        metadata['column_tokens'] = column_tokens
        metadata['sentence_tokens'] = sentence_tokens
        metadata['sqls'] = sqls
        fields['yesno_list'] = ListField(
            [LabelField(yesno, label_namespace="yesno_labels") for yesno in yesno_list])
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def find_passage_token_index(self, p_tokens, s_tokens, v_tokens):
        ## 先判断句子位置，再判断具体词位置
        ps, pe = self.find_idx(p_tokens, s_tokens)
        ss, se = self.find_idx(s_tokens, v_tokens)
        if ps == None or  pe == None or ss == None or se == None:
           return self.find_idx(p_tokens, v_tokens)
        return ps + ss, ps + se

    def find_idx(self, mylist, pattern):
        s, e = (None, None)
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                s = i
                e = i + len(pattern)
                break
        return s, e
