mvtec_path = "/mnt/data-server-storage/okatani_lab/mvtec/"
# "cait_m48_448" or "deit_base_distilled_patch16_384" is supported.
backbone = "cait_m48_448"
device = "cuda:0"
batch_size = 32
nb_epoch = 2
learning_rate = 2e-4
warmup_epoch = 1
validate_per_epoch = 1

clamp = 0.15
clamp_activation = "ATAN"

weight_path = "test"
result_path = "test"