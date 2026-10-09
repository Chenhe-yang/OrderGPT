dataset_name = "census"
data_dir = "data/census/records"
expected_condition_mode = ""
out_dir = "out/ordergpt/census/mlp"
variable_permutation_seed = 0


seed = 1337
device = "auto"
dtype = "bfloat16"
compile_model = True
init_from = "scratch"
resume_checkpoint = ""
n_layer = 2
n_head = 4
n_embd = 128
condition_layers = 0
dropout = 0.0
bias = True
min_log_scale = -7.0
max_log_scale = 5.0

batch_size = 512
gradient_accumulation_steps = 1
max_iters = 50_000
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 1_000
lr_decay_iters = 50_000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

log_interval = 20
eval_interval = 500
eval_batches = 16
test_eval_batches = 241
eval_only_split = "test"
always_save_checkpoint = True
best_selection_start_iter = 20_000
eval_only = False


method = "mlp"


attention_window_samples = 1_024


mlp_seed_offset = 3_000_003
mlp_hidden_dims = (32, 32)
mlp_learning_rate = 2e-4
mlp_weight_decay = 0.01
mlp_beta1 = 0.9
mlp_beta2 = 0.999
mlp_grad_clip = 1.0

wandb_log = False
wandb_project = "OrderGPT-CENSUS"
wandb_run_name = "census-mlp-seed1337"

distill_start = 5000
distill_end = 10000
anneal_end = 20000
