import math

from torch.utils.data import SequentialSampler

from f5_tts.model.dataset import DynamicBatchSampler, load_dataset


train_dataset = load_dataset("Emilia_ZH_EN", "pinyin")
sampler = SequentialSampler(train_dataset)

gpus = 8
batch_size_per_gpu = 38400
max_samples_per_gpu = 64
max_updates = 1250000

batch_sampler = DynamicBatchSampler(
    sampler,
    batch_size_per_gpu,
    max_samples=max_samples_per_gpu,
    random_seed=666,
    drop_residual=False,
)

print(
    f"One epoch has {len(batch_sampler) / gpus} updates if gpus={gpus}, with "
    f"batch_size_per_gpu={batch_size_per_gpu} (frames) & "
    f"max_samples_per_gpu={max_samples_per_gpu}."
)
print(
    f"If gpus={gpus}, for max_updates={max_updates} "
    f"should set epoch={math.ceil(max_updates / len(batch_sampler) * gpus)}."
)
