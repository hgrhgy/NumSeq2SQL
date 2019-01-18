import logging
import torch
from typing import Optional, Dict, List, Any
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, SquadEmAndF1, Average
from allennlp.nn import InitializerApplicator, util
from allennlp.modules.input_variational_dropout import InputVariationalDropout
import json
import numpy as np
from torch.nn.functional import nll_loss

from model.layers import FusionLayer, BilinearSeqAtt, FeedForward

logger = logging.getLogger(__name__)


@Model.register("numseq2sql")
class NumSeq2SQL(Model):

    def __init__(self, vocab: Vocabulary,
                 embedder: PretrainedBertEmbedder,
                 passage_BiLSTM: Seq2SeqEncoder,
                 columns_BiLSTM: Seq2SeqEncoder,
                 sentence_BiLSTM: Seq2SeqEncoder,
                 passage_contextual: Seq2SeqEncoder,
                 columns_contextual: Seq2SeqEncoder,
                 sentence_contextual: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(NumSeq2SQL, self).__init__(vocab, regularizer)

        # BERT Embedding
        self._embedder = embedder
        self._baidu_embedder = self.load_word_emb()

        self._passage_BiLSTM = passage_BiLSTM
        self._columns_BiLSTM = columns_BiLSTM
        self._sentence_BiLSTM = sentence_BiLSTM

        self._passage_contextual = passage_contextual
        self._columns_contextual = columns_contextual
        self._sentence_contextual = sentence_contextual

        self._encoding_dim = self._passage_BiLSTM.get_output_dim()
        self.projected_layer = torch.nn.Linear(self._encoding_dim, self._encoding_dim)
        self.fuse_p = FusionLayer(self._encoding_dim)
        self.fuse_c = FusionLayer(self._encoding_dim)
        self.fuse_s = FusionLayer(self._encoding_dim)

        self.linear_self_align = torch.nn.Linear(self._encoding_dim, 1)

        self.bilinear_layer_s = BilinearSeqAtt(self._encoding_dim, self._encoding_dim)
        self.bilinear_layer_e = BilinearSeqAtt(self._encoding_dim, self._encoding_dim)
        self.yesno_predictor = torch.nn.Linear(self._encoding_dim, 3)
        self.relu = torch.nn.ReLU()

        self._max_span_length = 30

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._variational_dropout = InputVariationalDropout(dropout)
        self._span_yesno_accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                column: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                col_start_idx: torch.IntTensor = None,
                col_end_idx: torch.IntTensor = None,
                val_start_idx: torch.IntTensor = None,
                val_end_idx: torch.IntTensor = None,
                yesno_list: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        ## 字数
        batch_size, max_sent_count, max_sent_len = sentence['bert'].size()

        ## 中文分词Token数
        _, _, max_sent_token_len = sentence['bert-offsets'].size()

        # # total_qa_count * max_q_len * encoding_dim
        total_sent_count = batch_size * max_sent_count
        yesno_mask = torch.ge(yesno_list, 0).view(total_sent_count)

        # embedded_question = embedded_question.reshape(total_qa_count, max_q_len, self._text_field_embedder.get_output_dim())
        embedded_sentence = self._embedder(sentence['bert']).reshape(total_sent_count, max_sent_len,
                                                                     self._embedder.get_output_dim())
        embedded_passage = self._embedder(passage['bert'])
        embedded_column = self._embedder(column['bert'])


        sentence_mask = util.get_text_field_mask(sentence, num_wrapping_dims=1).float().squeeze(1)

        # sentence_mask = sentence_mask.reshape(total_sent_count, max_sent_len - 2)
        # sentence_mask = sentence_mask.reshape(total_sent_count, max_sent_len)
        # sentence_mask = sentence_mask.new_ones(batch_size, max_sent_count, max_sent_len)
        # sentence_mask = [[[1] + s + [1]] for s in sentence_mask]
        column_mask = util.get_text_field_mask(column).float()

        # column_mask = column_mask.reshape(total_sent_count, max_sent_len)
        # column_mask = column_mask.new_ones(batch_col_size, max_col_count, max_col_len)
        passage_mask = util.get_text_field_mask(passage).float()

        encode_passage = self._passage_BiLSTM(embedded_passage, passage_mask)
        encode_sentence = self._sentence_BiLSTM(embedded_sentence, sentence_mask)
        encode_column = self._columns_BiLSTM(embedded_column, column_mask)

        passage_length = encode_passage.size(1)
        column_length = encode_column.size(1)

        projected_passage = self.relu(self.projected_layer(encode_passage))
        projected_sentence = self.relu(self.projected_layer(encode_sentence))
        projected_column = self.relu(self.projected_layer(encode_column))

        encoded_passage = self._variational_dropout(projected_passage)
        encode_sentence = self._variational_dropout(projected_sentence)
        encode_column = self._variational_dropout(projected_column)

        # repeated_encode_column = encode_column.repeat(1, max_col_count, 1, 1)

        repeated_encoded_passage = encoded_passage.unsqueeze(1).repeat(1, max_sent_count, 1, 1)
        repeated_encoded_passage = repeated_encoded_passage.view(total_sent_count, passage_length, self._encoding_dim)

        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, max_sent_count, 1)
        repeated_passage_mask = repeated_passage_mask.view(total_sent_count, passage_length)


        repeated_encode_column = encode_column.unsqueeze(1).repeat(1, max_sent_count, 1, 1)
        repeated_encode_column = repeated_encode_column.view(total_sent_count, column_length, self._encoding_dim)

        repeated_column_mask = column_mask.unsqueeze(1).repeat(1, max_sent_count, 1)
        repeated_column_mask = repeated_column_mask.view(total_sent_count, column_length)

        ## S2C
        s = torch.bmm(encode_sentence, repeated_encode_column.transpose(2, 1))
        alpha = util.masked_softmax(s, sentence_mask.unsqueeze(2).expand(s.size()), dim=1)
        aligned_s2c = torch.bmm(alpha.transpose(2, 1), encode_sentence)

        ## P2C
        p = torch.bmm(repeated_encoded_passage, repeated_encode_column.transpose(2, 1))
        beta = util.masked_softmax(p, repeated_passage_mask.unsqueeze(2).expand(p.size()), dim=1)
        aligned_p2c = torch.bmm(beta.transpose(2, 1), repeated_encoded_passage)

        ## C2S
        alpha1 = util.masked_softmax(s, repeated_column_mask.unsqueeze(1).expand(s.size()), dim=1)
        aligned_c2s = torch.bmm(alpha1, repeated_encode_column)

        ## C2P
        beta1 = util.masked_softmax(p, repeated_column_mask.unsqueeze(1).expand(p.size()), dim=1)
        aligned_c2p = torch.bmm(beta1, repeated_encode_column)

        fused_p = self.fuse_p(repeated_encoded_passage, aligned_c2p)
        fused_s = self.fuse_s(encode_sentence, aligned_c2s)
        fused_c = self.fuse_c(aligned_p2c, aligned_s2c)

        contextual_p = self._passage_contextual(fused_p, repeated_passage_mask)
        contextual_s = self._sentence_contextual(fused_s, sentence_mask)
        contextual_c = self._columns_contextual(fused_c, repeated_column_mask)

        contextual_c2p = torch.bmm(contextual_p, contextual_c.transpose(1, 2))
        alpha2 = util.masked_softmax(contextual_c2p, repeated_column_mask.unsqueeze(1).expand(contextual_c2p.size()), dim=1)
        aligned_contextual_c2p = torch.bmm(alpha2, contextual_c)

        contextual_c2s = torch.bmm(contextual_s, contextual_c.transpose(1, 2))
        beta2 = util.masked_softmax(contextual_c2s, repeated_column_mask.unsqueeze(1).expand(contextual_c2s.size()), dim=1)
        aligned_contextual_c2s = torch.bmm(beta2, contextual_c)

        # cnt * m
        gamma = util.masked_softmax(self.linear_self_align(aligned_contextual_c2s).squeeze(2), sentence_mask, dim=1)
        # cnt * h
        weighted_s = torch.bmm(gamma.unsqueeze(1), aligned_contextual_c2s).squeeze(1)

        # weighted_s = torch.bmm(gamma_s.unsqueeze(1), contextual_c2s).squeeze(1)

        span_start_logits = self.bilinear_layer_s(weighted_s, aligned_contextual_c2p)
        span_end_logits = self.bilinear_layer_e(weighted_s, aligned_contextual_c2p)

        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

        span_yesno_logits = self.yesno_predictor(torch.bmm(span_end_logits.unsqueeze(2), weighted_s.unsqueeze(1)))

        best_span = self._get_best_span(span_start_logits, span_end_logits,span_yesno_logits,
                                        self._max_span_length)
        output_dict: Dict[str, Any] = {}

        # Compute the loss for training

        if col_start_idx is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, repeated_passage_mask), col_start_idx.view(-1),
                            ignore_index=-1)
            self._span_start_accuracy(span_start_logits, col_start_idx.view(-1), mask=yesno_mask)
            loss += nll_loss(util.masked_log_softmax(span_end_logits, repeated_passage_mask), col_end_idx.view(-1),
                             ignore_index=-1)
            self._span_end_accuracy(span_end_logits, col_end_idx.view(-1), mask=yesno_mask)
            self._span_accuracy(best_span[:, 0:2],
                                torch.stack([col_start_idx, col_end_idx], -1).view(total_sent_count, 2),
                                mask=yesno_mask.unsqueeze(1).expand(-1, 2).long())
            gold_span_end_loc = []
            col_end_idx = col_end_idx.view(total_sent_count).squeeze().data.cpu().numpy()
            for i in range(0, total_sent_count):
                # print(total_sent_count)

                gold_span_end_loc.append(max(col_end_idx[i] * 3 + i * passage_length * 3, 0))
                gold_span_end_loc.append(max(col_end_idx[i] * 3 + i * passage_length * 3 + 1, 0))
                gold_span_end_loc.append(max(col_end_idx[i] * 3 + i * passage_length * 3 + 2, 0))
            gold_span_end_loc = col_start_idx.new(gold_span_end_loc)
            pred_span_end_loc = []
            for i in range(0, total_sent_count):
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3, 0))
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 1, 0))
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 2, 0))
            predicted_end = col_start_idx.new(pred_span_end_loc)

            _yesno = span_yesno_logits.view(-1).index_select(0, gold_span_end_loc).view(-1, 3)
            loss += nll_loss(torch.nn.functional.log_softmax(_yesno, dim=-1), yesno_list.view(-1), ignore_index=-1)

            _yesno = span_yesno_logits.view(-1).index_select(0, predicted_end).view(-1, 3)
            self._span_yesno_accuracy(_yesno, yesno_list.view(-1), mask=yesno_mask)
            output_dict["loss"] = loss


        output_dict['best_span_str'] = []
        output_dict['qid'] = []
        best_span_cpu = best_span.detach().cpu().numpy()
        for i in range(batch_size):
            passage_str = metadata[i]['origin_passage']
            offsets = passage['bert-offsets'][i].cpu().numpy()
            f1_score = 0.0
            per_dialog_best_span_list = []
            per_dialog_query_id_list = []
            for per_dialog_query_index, sql in enumerate(metadata[i]["sqls"]):

                predicted_span = tuple(best_span_cpu[i * max_sent_count + per_dialog_query_index])
                start_offset = offsets[predicted_span[0]]
                end_offset = offsets[predicted_span[1]]
                per_dialog_query_id_list.append(sql)
                best_span_string = ''.join([ t.text for t in metadata[i]['passage_tokens'][start_offset:end_offset]])
                #print(best_span_string)
                per_dialog_best_span_list.append(best_span_string)

            output_dict['qid'].append(per_dialog_query_id_list)
            output_dict['best_span_str'].append(per_dialog_best_span_list)
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        ## do nothing
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset)}

    @staticmethod
    def _get_best_span(span_start_logits: torch.Tensor,
                       span_end_logits: torch.Tensor,
                       span_yesno_logits: torch.Tensor,
                       max_span_length: int) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 3), dtype=torch.long)
        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        span_yesno_logits = span_yesno_logits.data.cpu().numpy()

        for b_i in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                if val1 < span_start_logits[b_i, j]:
                    span_start_argmax[b_i] = j
                    val1 = span_start_logits[b_i, j]
                val2 = span_end_logits[b_i, j]
                if val1 + val2 > max_span_log_prob[b_i]:
                    if j - span_start_argmax[b_i] > max_span_length:
                        continue
                    best_word_span[b_i, 0] = span_start_argmax[b_i]
                    best_word_span[b_i, 1] = j
                    max_span_log_prob[b_i] = val1 + val2
        #
        for b_i in range(batch_size):
            j = best_word_span[b_i, 1]
            yesno_pred = np.argmax(span_yesno_logits[b_i, j])
            best_word_span[b_i, 2] = int(yesno_pred)
        return best_word_span

    def load_word_emb(self):
        with open('baidu/word2idx.json', encoding='UTF-8') as inf:
            w2i = json.load(inf)
        with open('baidu/usedwordemb.npy', "rb") as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
