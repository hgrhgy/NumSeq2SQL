## NumSeq2SQL

### Tutorial
First you should install allennlp and make sure you have downloaded bert pretrained model. </br>
bert model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz </br>
bert vocab: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt </br>
</br>
for train, plean run: </br>
`allennlp train ./config/numseq2sql.json -s target --include-package numseq2sql` </br>
To modified the parameters for the model, you can see `config/numseq2sql.json` </br>

### Next
employ Chinese Tokenization

### Main Reference
SQLNet: [https://github.com/xiaojunxu/SQLNet](https://github.com/xiaojunxu/SQLNet) </br>
SLQA: [https://github.com/SparkJiao/SLQA](https://github.com/SparkJiao/SLQA) </br>
