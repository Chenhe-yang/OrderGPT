dataset_name = "h3wb"
data_dir = "data/h3wb/processed"
expected_condition_mode = "official_i2d"
out_dir = "out/ordergpt/h3wb/mlp"
variable_permutation_seed = 0


seed = 1340
device = "auto"
dtype = "bfloat16"
compile_model = False
init_from = "scratch"
resume_checkpoint = ""
n_layer = 4
n_head = 8
n_embd = 256
condition_layers = 1
dropout = 0.0
bias = True
min_log_scale = -7.0
max_log_scale = 5.0

batch_size = 160
gradient_accumulation_steps = 1
max_iters = 100_000
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 5_000
lr_decay_iters = 100_000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

log_interval = 20
eval_interval = 1_000
eval_batches = 13
test_eval_batches = 0
eval_only_split = "val"
always_save_checkpoint = True
best_selection_start_iter = 60_000
eval_only = False


method = "mlp"


attention_window_samples = 1_280


mlp_seed_offset = 3_000_003
mlp_hidden_dims = (32, 32)
mlp_learning_rate = 2e-4
mlp_weight_decay = 0.01
mlp_beta1 = 0.9
mlp_beta2 = 0.999
mlp_grad_clip = 1.0

wandb_log = False
wandb_project = "OrderGPT-H3WB"
wandb_run_name = "h3wb-mlp-seed1340"

distill_start = 15000
distill_end = 40000
anneal_end = 60000
