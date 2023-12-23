### FMCF 
This paper proposes a fusing multiple code features (FMCF) approach based on Transformer for Solidity smart Contracts. 
This approach can extract and fuse the SBT sequence features and AST features , improving the quality of contract code summarization.
### Datasets
- The smart contract corpus we use comes from this website:
- https://zenodo.org/record/4587089#.YEMmWugzYuU.
- https://github.com/NTDXYG/CCGIR
### Data preprocessing
#### AST
- The Solidity smart contract corpus code was converted to AST using the recognized Solidity Parser Antlr parsing package.https://github.com/federicobond/solidity-parser-antlr
#### CodeBERT
- Installation dependencies: First, ensure that you have installed the necessary software and libraries, including Python, PyTorch, transformers, and tokenizers. You can use pip or conda for installation.
- Load pre trained model: Select a CodeBERT pre trained model suitable for your task from the [Hugging Face](https://huggingface.co/microsoft/codebert-base) model library, such as Microsoft/codebert base. Here, AutoModel and AutoToken are used to load the model and tokenizer.
- Alternatively, using the BERT client server mode, the pre trained model can be downloaded to the local machine.
```
pip install bert-serving-server  
pip install bert-serving-client
```
### Requirements
```
Python 3.8
bert-serving-client 1.10.0  
bert-serving-server 1.10.0   
nltk 3.5
rough 1.0
```
