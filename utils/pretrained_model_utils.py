from networks.autoencoder_modules import MovementConvDecoder, MovementConvEncoder
from utils.paramUtils import DIMPOSE
import torch
from os.path import join as pjoin
from sentence_transformers import SentenceTransformer


def get_pretrained_vae(checkpoint_dir):
    encoder = MovementConvEncoder(
        input_size = DIMPOSE - 4,
        hidden_size = 512,
        output_size = 512
    )
    decoder = MovementConvDecoder(
        input_size = 512,
        hidden_size = 512,
        output_size = DIMPOSE - 4
    )

    humanml3d_vae_chkpoint = torch.load(pjoin(checkpoint_dir, 'model/humanml3d_pretrained_vae.tar'), map_location = torch.device("cpu"))

    encoder.load_state_dict(humanml3d_vae_chkpoint['movement_enc'])
    decoder.load_state_dict(humanml3d_vae_chkpoint['movement_dec'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def get_pretrained_text_encoder(device):
    text_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device = str(device)
    )
    text_encoder.eval()

    return text_encoder