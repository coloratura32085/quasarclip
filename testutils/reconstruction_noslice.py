# Set up SpecFormer model and move to the correct device
import datasets
import numpy as np
import torch
from torch import device
import matplotlib.pyplot as plt

from models.specformer_no_slice import SpecFormer
from root_path import ROOT_PATH
from task.get_embed import remove_prefix

pretrained_weights = {
    "specformer": '/home/kongxiao/data45/kx/dongbixin/outputs/spec/spec_g3z_no_slice_0mask/logs/lightning_logs/version_0/checkpoints/last.ckpt',
    # "vitae": '/mnt/d/database/img_1w/output/pths/_64/2000_0.75_0.001_0.05_64/best_loss.pth',
    # "specformer": '/mnt/d/database/astroclip/last.ckpt',
}

checkpoint = torch.load(pretrained_weights["specformer"])
params = checkpoint["hyper_parameters"]
del params['lr'],params['weight_decay'],params['T_max'],params['T_warmup']
specformer = SpecFormer(**params)
state_dict = remove_prefix(checkpoint["state_dict"], 'model.')
specformer.load_state_dict(state_dict,strict=False)
# device = torch.device("cpu")
# specformer.to(device)
# print(specformer)

data = datasets.load_from_disk(f'{ROOT_PATH}/data/data_gaussian/train_dataset')
spectra = data['spectrum']
spectra = spectra.clone().detach().unsqueeze(-1)[:,0 : 3900, :]
print('spectra.shape',spectra.shape)
# 确保 spectra 的长度大于或等于 10
if len(spectra) >= 10:
    indices = np.random.choice(len(spectra), size=10, replace=False)
    spectra = spectra[indices]
else:
    raise ValueError("The number of spectra is less than 10.")
input = specformer.preprocess(spectra)
input = input.detach().numpy()
input = input[:,:,2:]
input = input.reshape(input.shape[0],input.shape[2],1)
print('input.shape' ,input.shape)
output = specformer(spectra)['reconstructions']
output = output.detach().numpy()
output = output[:,:,2:]
output = output.reshape(output.shape[0],output.shape[2],1)
print('out.shape',output.shape)
# def unshape(x):
#     out = []
#     for j in range(x.shape[0]):
#         t = x[j,:,:]
#         result = []
#         # 转换为 NumPy 数组
#         t = np.array(t)
#         # print(t.shape)
#         result.append(t[ 0, 0:10])
#         for i in range(1,t.shape[0] -1 ):
#             overlap = (t[ i, 10:] + t[ i + 1, 0:10])/2
#             result.append(overlap)
#         result.append(t[t.shape[0] -1 , 10:])
#         result = np.array(result).flatten()
#         out.append(result)
#     return out
#
# output = unshape(output)
# input = unshape(input)
#

plt.figure(figsize=(12, 6))
plt.plot(input[2], label='Input', color='blue')
plt.plot(output[2], label='Output', color='red', linestyle='--')
plt.title('Comparison of Input and Output Spectra')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()