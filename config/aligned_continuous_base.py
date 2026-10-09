dataset_name = ""
data_dir = ""

expected_condition_mode = ""

out_dir = "out/aligned_continuous/base"
variable_permutation_seed = -1


seed = 1337
device = "auto"
dtype = "bfloat16"
compile_model = False
init_from = "scratch"
resume_checkpoint = ""

n_layer = 4
n_head = 4
n_embd = 256
condition_layers = 1
dropout = 0.0
bias = True
min_log_scale = -7.0
max_log_scale = 5.0

batch_size = 128
gradient_accumulation_steps = 1
max_iters = 20000
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 500
lr_decay_iters = 20000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

log_interval = 20
eval_interval = 250
eval_batches = 50
test_eval_batches = 0
eval_only_split = "val"
always_save_checkpoint = False


best_selection_start_iter = 0
eval_only = False


method = "teacher"





attention_window_samples = 1024


mlp_seed_offset = 3000003
mlp_hidden_dims = (32, 32)
mlp_learning_rate = 2e-4
mlp_weight_decay = 0.01
mlp_beta1 = 0.9
mlp_beta2 = 0.999
mlp_grad_clip = 1.0

wandb_log = False
wandb_mode = "online"
wandb_project = "OrderGPT"
wandb_run_name = ""

distill_start = 5000
distill_end = 10000
anneal_end = 20000
