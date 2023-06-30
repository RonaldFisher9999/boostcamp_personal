```bash
conda init
conda create -n seq python=3.9 -y
conda activate seq
pip install -r requirements.txt
```


- Basic configuration (`config.json`)
  - max_len: Maximum sequence length. default=200
  - n_samples: Number of sample sequences per user. default=10
  - tail_ratio: Ratio of tail items in sampled sequences. default=0.5
  - k: Number of recommendations. default=10
  - mask_prob: Masking probabilty. default=0.4
  - embed_dim: Dimension of embeddings. default=128
  - n_layers: Number of encoder layers. default=2
  - n_heads: Number of heads in attention. default=4
  - pffn_hidden_dim: Hidden dimension of point-wise feed forward network in a encoder layer. default=512
  - dropout_rate: Dropout rate. default=0.3
  - unidirection: Indicates if the model is unidirectional. default=false
  - batch_size: Batch size. default=256
  - lr: Learning rate. default=0.001
  - n_epochs: Number of epochs. default=100
  - max_patience: Patience epochs before stopping. default=15
  - data_dir: Directory of data. default="../../data/"
  - model_dir: Directory of model. default="./model/"
  - output_dir: Directory of output. default="./output/"
  - logging: Indicates if WandB logging is enabled. default=false
  - seed: Random seed for reproducibility. default=100


- Making inference with pretrained model (`pretrain.json`)
  - Configuration file(`model_name.json`) and state dictionary file(`model_name.pt`) should be located in the `model_dir` directory.
  - infer_k: Number of recommendation. default=10.
  - data_dir: Directory of data. default="../../data/"
  - model_dir: Directory of pretrained model and configuration file. default="./model/",
  - model_name: Name of the model. default="best"
  - output_dir: Directory of output. default="./output/"