import os
from argparse import ArgumentParser
from typing import Dict

import datasets

from imageutil.pos_embed import interpolate_pos_embed
from models.image_encoder import MaskedAutoencoderViTAE, mae_vitae_base_patch8_enc
import numpy as np
import torch
from astropy.table import Table
from tqdm import tqdm
from models.clip import AstroClipModel, ImageHead
from models.specformer import SpecFormer

# Set device for model (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def remove_prefix(state_dict, prefix):
    """
    移除权重前缀，方便加载模型。
    """
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def get_embeddings(
    image_models: Dict[str, torch.nn.Module],
    spectrum_models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    spectra: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    full_keys = set(image_models.keys()).union(spectrum_models.keys())
    model_embeddings = {key: [] for key in full_keys}
    im_batch, sp_batch = [], []

    assert len(images) == len(spectra)
    for image, spectrum in tqdm(zip(images, spectra)):
        # print("image.shape = ", image.shape)
        # print("spectrum.shape = ", spectrum.shape)
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])
        sp_batch.append(torch.tensor(spectrum, dtype=torch.float32)[None, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                # Ensure images and spectra are on the correct device
                spectra, images = torch.cat(sp_batch).to(device), torch.cat(im_batch).to(device)

                for key in image_models.keys():
                    model_embeddings[key].append(image_models[key](images))

                for key in spectrum_models.keys():
                    model_embeddings[key].append(spectrum_models[key](spectra))

            im_batch, sp_batch = [], []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
            spectra, images = torch.cat(sp_batch).to(device), torch.cat(im_batch).to(device)

            # Get embeddings
            for key in image_models.keys():
                model_embeddings[key].append(image_models[key](images))

            for key in spectrum_models.keys():
                model_embeddings[key].append(spectrum_models[key](spectra))

    model_embeddings = {
        key: np.concatenate(model_embeddings[key]) for key in model_embeddings.keys()
    }
    return model_embeddings


def embed_provabgs(
    provabgs_file_train: str,
    provabgs_file_test: str,
    batch_size: int = 512,
):


    pretrained_weights = {
        "astroclip": '/mnt/d/database/astroclip/outputs/astroclip/logs/lightning_logs/version_1/checkpoints/last.ckpt',
        "vitae": '/mnt/d/database/img_1w/output/pths/_64/2000_0.75_0.001_0.05_64/best_loss.pth',
        "specformer": '/mnt/d/database/astroclip/last.ckpt',
    }

    # Set up AstroCLIP model and move to the correct device
    astroclip = AstroClipModel.load_from_checkpoint(
        checkpoint_path=pretrained_weights["astroclip"], strict=False
    ).to(device)

    # Set up SpecFormer model and move to the correct device
    checkpoint = torch.load(pretrained_weights["specformer"])
    specformer = SpecFormer(**checkpoint["hyper_parameters"])
    state_dict = remove_prefix(checkpoint["state_dict"], 'model.')
    specformer.load_state_dict(state_dict)
    specformer.to(device)

    # Set up vitae model and move to the correct device
    vitae = mae_vitae_base_patch8_enc().to(device)
    checkpoint = torch.load(pretrained_weights['vitae'], map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = vitae.state_dict()

    # Remove unnecessary keys if shapes don't match
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(vitae, checkpoint_model)

    # Load pre-trained weights and freeze the parameters
    msg = vitae.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    for param in vitae.parameters():
        param.requires_grad = False

    # Set up model dict
    image_models = {
        "vitae": lambda x: np.mean(
            vitae(x).
            cpu().
            numpy(),axis=1
        ),
        "astroclip_image": lambda x: astroclip(x, input_type="image").cpu().numpy(),
    }

    spectrum_models = {
        "astroclip_spectrum": lambda x: astroclip(x, input_type="spectrum")
        .cpu()
        .numpy(),
        "specformer": lambda x: np.mean(
            specformer(x)["reconstruction"].cpu().numpy(), axis=1
        ),
    }
    print("Models are correctly set up!")

    embed = {}

    # Load data
    files = [provabgs_file_test, provabgs_file_train]
    for f in files:
        # provabgs = Table.read(f)
        provabgs = datasets.load_from_disk(f)
        images, spectra = provabgs["image"], provabgs['spectrum'].unsqueeze(-1)

        # Get embeddings
        embeddings = get_embeddings(
            image_models, spectrum_models, images, spectra, batch_size
        )

        # # Remove images and replace with embeddings
        # provabgs.remove_column("image")
        # provabgs.remove_column("spectrum")
        for key in embeddings.keys():
            assert len(embeddings[key]) == len(provabgs), "Embeddings incorrect length"
            embed[f"{key}_embeddings"] = embeddings[key]
        for key in embed.keys():
            tmp = f.split('/')[-1].split('_'[0])
            np.save(f"{tmp}_{key}.npy", embed[key])
            print(key)
            print(type(embed[key]))
            print(embed[key].shape)

        # Save embeddings
        # provabgs.write(f.replace(".hdf5", "_embeddings.hdf5"), overwrite=True)


if __name__ == "__main__":
    ASTROCLIP_ROOT = '/mnt/f/database/filterWave/data_64_mini'
    parser = ArgumentParser()
    parser.add_argument(
        "--provabgs_file_train",
        type=str,
        default=f"../data/data_g3_z/train_dataset",
    )
    parser.add_argument(
        "--provabgs_file_test",
        type=str,
        default=f"./data/data_g3_z/test_dataset",
    )

    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    embed_provabgs(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.batch_size,
    )
