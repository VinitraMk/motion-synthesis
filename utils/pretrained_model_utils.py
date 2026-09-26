from networks.autoencoder_modules import MovementConvDecoder, MovementConvEncoder
from networks.nn import MotionVAE
from utils.paramUtils import DIMPOSE
import torch
from os.path import join as pjoin
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, CLIPTextModel


def get_pretrained_vae(model_dir, meta_dir, max_motion_length):
    encoder = MovementConvEncoder(
        input_size = DIMPOSE - 4,
        hidden_size = 512,
        output_size = 512
    )
    decoder = MovementConvDecoder(
        input_size = 512,
        hidden_size = 512,
        output_size = DIMPOSE
    )
    motionvae = MotionVAE(
        dim = 263, #input dimension of motion vector
        hidden_size = 256, # latent dimension
        max_seq_len=max_motion_length,
        num_heads = 4,
        depth = 9,
        meta_dir = meta_dir,
        enable_skip_connections=True,
        t_latent = 6
    )

    humanml3d_vae_chkpoint = torch.load(pjoin(model_dir, 'humanml3d_pretrained_vae.tar'), map_location = torch.device("cpu"))
    motionvae_chkpoint = torch.load(pjoin(model_dir, 'motionvae_debug_d9_t6.tar'), map_location = torch.device("cpu"))

    encoder.load_state_dict(humanml3d_vae_chkpoint['movement_enc'])
    decoder.load_state_dict(humanml3d_vae_chkpoint['movement_dec'])
    motionvae.load_state_dict(motionvae_chkpoint['vae'])

    encoder.eval()
    decoder.eval()
    motionvae.eval()

    return encoder, decoder, motionvae

def get_pretrained_text_encoder(model:str = 'sentence_transformer', device = torch.device("cpu")):
    if model == 'clip_text':
        model_id = "openai/clip-vit-large-patch16"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
    else:
        text_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device = str(device)
        )
        text_encoder.eval()
        tokenizer = None

    return text_encoder, tokenizer
