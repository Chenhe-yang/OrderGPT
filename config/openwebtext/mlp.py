method = "mlp"
dataset_name = "openwebtext"
data_dir = "data/openwebtext"
out_dir = "out/ordergpt/openwebtext/mlp"
init_from = "scratch"
resume_checkpoint = ""


variable_permutation_seed = 42
variable_permutation_backend = "torch_randperm"
seed = 2027

block_size = 1024
vocab_size = 50304
n_layer = 12
n_head = 10
n_embd = 640
dropout = 0.0
bias = False


batch_size = 40
gradient_accumulation_steps = 2
expected_world_size = 6
max_iters = 400_000
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2_000
lr_decay_iters = 400_000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


attention_window_samples = 2_400


mlp_hidden_dims = (32, 32)
mlp_learning_rate = 2e-4
mlp_weight_decay = 0.01
mlp_beta1 = 0.9
mlp_beta2 = 0.99
mlp_grad_clip = 1.0
mlp_seed_offset = 26

eval_interval = 2_000
log_interval = 10
wandb_log = False
wandb_project = "OrderGPT-OpenWebText-400k"
wandb_run_name = "openwebtext-seq1024-ordergpt-mlp"

device = "cuda"
dtype = "bfloat16"
compile_model = True
ddp_backend = "nccl"

distill_start = 4000
distill_end = 40000
anneal_end = 60000
