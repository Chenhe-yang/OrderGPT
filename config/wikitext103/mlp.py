method = "mlp"
dataset_name = "wikitext103"
data_dir = "data/wikitext103"
out_dir = "out/ordergpt/wikitext103/mlp"
init_from = "scratch"
resume_checkpoint = ""
always_save_checkpoint = True

variable_permutation_seed = 42
variable_permutation_backend = "numpy_default_rng"
seed = 2053

block_size = 80
vocab_size = 50304
n_layer = 4
n_head = 8
n_embd = 384
dropout = 0.0
bias = False

batch_size = 256
gradient_accumulation_steps = 2
max_iters = 50_000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 0
lr_decay_iters = 50_000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0


attention_window_samples = 1024

mlp_hidden_dims = (32, 32)
mlp_learning_rate = 2e-4
mlp_weight_decay = 0.01
mlp_beta1 = 0.9
mlp_beta2 = 0.99
mlp_grad_clip = 1.0
mlp_seed_offset = 3_000_003

eval_interval = 250
eval_iters = 200
eval_batch_size = 64
log_interval = 10
wandb_log = False
wandb_project = "OrderGPT-WikiText103"
wandb_run_name = "wikitext103-seq80-ordergpt-mlp"

device = "cuda"
dtype = "bfloat16"
compile_model = True
ddp_backend = "nccl"

distill_start = 10000
distill_end = 18000
anneal_end = 32000
