from networks.autoencoder_modules import MovementConvDecoder, MovementConvEncoder
from utils.paramUtils import DIMPOSE
import torch
from os.path import join as pjoin
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPTextModelWithProjection, CLIPTextModel


def get_pretrained_vae(model_dir):
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

    humanml3d_vae_chkpoint = torch.load(pjoin(model_dir, 'humanml3d_pretrained_vae.tar'), map_location = torch.device("cpu"))

    encoder.load_state_dict(humanml3d_vae_chkpoint['movement_enc'])
    decoder.load_state_dict(humanml3d_vae_chkpoint['movement_dec'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def get_pretrained_text_encoder(model:str = 'sentence_transformer', device = torch.device("cpu")):
    if model == 'clip_text':
        model_id = "openai/clip-vit-large-patch14"
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
