import os
import argparse
from pathlib import Path
from os.path import join as pjoin
import torch
from networks.nn import DiT
from networks.transformer_modules import TextTokenEncoder
from utils.pretrained_model_utils import get_pretrained_vae, get_pretrained_text_encoder
import numpy as np
import json
from data_utils.motion_processor import recover_from_ric, HUMANML3D_SKELETON_EDGES, _draw_skeleton_frame, _set_equal_3d_axes
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

class MotionPipeline:
    def __init__(self, dit: DiT, text_embedder, text_tokenizer, decoder, device, meta_dir,
        num_train_timesteps=1000,
        num_inference_steps = 1000,
        beta_start=1e-4,
        beta_end=2e-2, prediction_type = "epsilon"):

        self.dit = dit
        self.dit.eval()
        self.text_embedder = text_embedder
        self.text_tokenizer = text_tokenizer
        self.decoder = decoder
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type

        self.mean = np.load(pjoin(meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(meta_dir, 'std.npy'))

        with open(pjoin(meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.part_names = mapping['part_names']
        self.d_part_max = mapping['d_part_max']
        self.joints_num = mapping['joints_num']
        self.build_schedule(self.beta_start, self.beta_end, self.num_train_timesteps)


    def denormalize_motion(self, motion):
        return motion * self.std + self.mean
    

    def build_schedule(self, beta_start, beta_end, num_train_timesteps):
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32, device=self.device), alphas_cumprod[:-1]],
            dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def _p_sample(self, x, t, text_embeddings, text_mask=None):
        B = x.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
        d = torch.zeros_like(t_batch)

        model_pred = self.dit(x, t_batch, d, text_embeddings, text_mask)

        if self.prediction_type == "epsilon":
            betas_t = self._extract(self.betas, t_batch, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
            )
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t_batch, x.shape)

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_pred / sqrt_one_minus_alphas_cumprod_t
            )
        elif self.prediction_type == "x0":
            betas_t = self._extract(self.betas, t_batch, x.shape)
            alphas_t = self._extract(self.alphas, t_batch, x.shape)
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t_batch, x.shape)

            coef1 = betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            coef2 = (1.0 - alphas_cumprod_prev_t) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod_t)
            model_mean = coef1 * model_pred + coef2 * x
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t_batch, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def _sample(self, text_embeddings, seq_len, latent_dim, text_mask=None, batch_size=1):
        self.dit.eval()

        text_embeddings= text_embeddings.to(self.device)
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
        print('text metrics', text_embeddings.mean().item(), text_embeddings.std().item())
        x = torch.randn(batch_size, seq_len, latent_dim, device=self.device)
        full_t = self.num_train_timesteps
        timesteps = np.linspace(0, full_t - 1, self.num_inference_steps)
        timesteps = list(np.round(timesteps).astype(int))

        for t in reversed(timesteps):
            x = self._p_sample(x, t, text_embeddings, text_mask)

        return x
    
    @torch.no_grad()
    def __calltest__(self, prompt, generator, num_inference_steps=50, seed=42, latent_shape=(1, 512), eta = 0.0):
        #generator = torch.Generator(device=self.device).manual_seed(seed)

        z1 = torch.randn(latent_shape, generator=generator, device=self.device)
        z2 = z1.clone()

        #cond = self.text_embedder.encode(prompt, convert_to_tensor = True, device = str(self.device)).clone()
        inputs = self.text_tokenizer(["a person walking"], return_tensors="pt", padding="max_length", truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        text_embeddings = self.text_embedder(**inputs).last_hidden_state
        text_mask = inputs['attention_mask']
        inputs1 = self.text_tokenizer(["a person jumping"], return_tensors="pt", padding="max_length", truncation=True)
        inputs1 = {name: tensor.to(self.device) for name, tensor in inputs1.items()}
        text_embeddings1 = self.text_embedder(**inputs1).last_hidden_state
        text_mask1 = inputs1['attention_mask']
        #text_embeddings, text_mask = self.text_embedder.encode_tokens(["a person walking"])
        #text_embeddings1, text_mask1 = self.text_embedder.encode_tokens(["a person jumping"])
        print(prompt, num_inference_steps)
        print(text_embeddings.mean().item(), text_embeddings.std().item())
        print(text_embeddings1.mean().item(), text_embeddings1.std().item())

        latent = self._sample(
            text_embeddings = text_embeddings,
            seq_len=120//4,
            latent_dim = 512,
            text_mask = text_mask,
            batch_size = 1
        )
        latent1 = self._sample(
            text_embeddings = text_embeddings1,
            seq_len=120//4,
            latent_dim = 512,
            text_mask = text_mask1,
            batch_size = 1
        )
        print('latent stats: ', latent.mean(), latent.std(), latent.abs().max())
        print('latent1 stats: ', latent1.mean(), latent1.std(), latent1.abs().max())

        motion1 = self.decoder(latent)
        motion2 = self.decoder(latent1)
        denormalized_motion1 = self.denormalize_motion(motion1[0])
        denormalized_motion2 = self.denormalize_motion(motion2[0])

        motion_joints1 = recover_from_ric(denormalized_motion1, self.joints_num)
        motion_joints2 = recover_from_ric(denormalized_motion2, self.joints_num)

        return motion_joints1, motion_joints2
    
    @torch.no_grad()
    def __callshorcut__(self, prompt, generator, num_inference_steps=50, seed=42, latent_shape=(1, 512), eta = 0.0):
        #generator = torch.Generator(device=self.device).manual_seed(seed)

        z = torch.randn(latent_shape, generator=generator, device=self.device)

        #cond = self.text_embedder.encode(prompt, convert_to_tensor = True, device = str(self.device)).clone()
        text_tokens, text_mask = self.text_embedder.encode_tokens([prompt])
        #print(prompt, num_inference_steps)
        #print(text_tokens.mean().item(), text_tokens.std().item())

        for d_step in range(num_inference_steps):
            t = torch.full((z.shape[0], ), fill_value = d_step / (num_inference_steps - 1), device = self.device)
            t = t.clamp(1e-4, 1.0)
            d = torch.zeros(z.shape[0], device=self.device)
            alpha = 1.0 / num_inference_steps

            v = self.dit(z, t, d, text_tokens, text_mask)
            z = z + alpha * v


        motion = self.decoder(z)
        print('generated motion shape: ', z.shape, motion.shape)
        denormalized_motion = self.denormalize_motion(motion[0])

        motion_joints = recover_from_ric(denormalized_motion, self.joints_num)

        return motion_joints
    
    @torch.no_grad()
    def __call__(self, prompt, max_motion_len, latent_dim = 512, batch_size = 1):
        inputs = self.text_tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        text_embeddings = self.text_embedder(**inputs).last_hidden_state
        text_mask = inputs['attention_mask']
        latent = self._sample(
            text_embeddings = text_embeddings,
            seq_len=max_motion_len//4,
            latent_dim = latent_dim,
            text_mask = text_mask,
            batch_size = batch_size
        )
        print('latent stats: ', latent.mean(), latent.std(), latent.abs().max())
        motion = self.decoder(latent)
        denormalized_motion = self.denormalize_motion(motion[0])
        motion_joints = recover_from_ric(denormalized_motion, self.joints_num)
        return motion_joints

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="A prompt for the generation pipeline", default = 'A person waving their arms')
    parser.add_argument("--output_dir", type=str, default="outputs/diffusion_inference")
    parser.add_argument("--model_dir", type=str, default="checkpoints/model")
    parser.add_argument("--meta_dir", type=str, default="checkpoints/HumanML3D/test/meta")
    parser.add_argument("--gif_name", type=str, default="sample.gif")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument('--max_motion_length', type = int, default = 120)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(device, model_dir, meta_dir, max_motion_length = 40):
    
    enc, dec = get_pretrained_vae(model_dir=model_dir)
    enc.eval()
    dec.eval()
    text_encoder, text_tokenizer = get_pretrained_text_encoder(model = "clip_text", device = device)
    #text_encoder = TextTokenEncoder(device = device).to(device)
    #text_tokenizer = None
    text_encoder.eval()
    dit_chkpt = torch.load(pjoin(model_dir, 'dit_stable_crossattn_full.tar'), map_location = device)
    dit = DiT(
        input_size = 512,
        hidden_size=1152,
        text_dim=768,
        max_seq_len=max_motion_length//4
    )
    dit.load_state_dict(dit_chkpt['dit'])
    dit.to(device)
    dit.eval()

    pipe = MotionPipeline(
        dit=dit,
        text_embedder=text_encoder,
        text_tokenizer=text_tokenizer,
        decoder=dec,
        device=device,
        meta_dir=meta_dir
    )

    return pipe, (enc, dec)


def render_skeleton_animation(
    joints_recon: np.ndarray,
    output_path_no_ext: str,
    clip_id: str,
    text = None,
    recon_caption: str = "",
    skeleton_edges=HUMANML3D_SKELETON_EDGES,
    fps: int = 20,
    save_gif_fallback: bool = True,
):
    if not(MATPLOTLIB_AVAILABLE):
        print("Matplotlib not available")
        return None
    
    joints_recon = np.asarray(joints_recon)

    num_frames = joints_recon.shape[0]
    joints_recon = joints_recon[:num_frames]
    recon_caption = recon_caption if recon_caption != "" else "Recon"
    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
        
    caption = text if text else clip_id
    fig.suptitle(caption, fontsize = 12, y = 0.98)
    xyz = joints_recon.reshape(-1, 3)

    def update(frame_idx):
        _draw_skeleton_frame(ax, joints_recon[frame_idx], skeleton_edges, f"{recon_caption} frame={frame_idx}")
        _set_equal_3d_axes(ax, xyz)
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, interval=max(1, int(1000 / fps)), blit=False)

    saved_path = None
    '''
    if save_mp4:
        try:
            mp4_path = output_path_no_ext + ".mp4"
            writer = FFMpegWriter(fps=fps, metadata={"artist": "vqvae-validator"})
            anim.save(mp4_path, writer=writer)
            saved_path = mp4_path
        except Exception as exc:
            print('file path: ', output_path_no_ext)
            print('Failed to save motion mp4: ', exc)
            saved_path = None
    '''

    if saved_path is None and save_gif_fallback:
        try:
            gif_path = output_path_no_ext + ".gif"
            writer = PillowWriter(fps=num_frames)
            anim.save(gif_path, writer=writer)
            saved_path = gif_path
        except Exception as exec:
            saved_path = None
            print('GIF file not saved: ', exec)
    plt.close(fig)
    return saved_path



@torch.no_grad()
def run_inference(pipe, prompt, num_steps, max_motion_len, seed, device, outputs_path):

    # Replace this call with your actual pipeline invocation
    #prompt = 'run'
    gen = torch.Generator(device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        max_motion_len = max_motion_len,
    )

    motion_joints = result
    all_gif_files = [fname for fname in os.listdir(outputs_path) if ".gif" in fname]
    file_id = len(all_gif_files)
    render_skeleton_animation(
        joints_recon=motion_joints,
        output_path_no_ext=pjoin(outputs_path, f'inference_test_clip_{file_id}'),
        clip_id=f'inference_test_clip_{file_id}',
        text = 'a person walking',
        recon_caption='Diffusion inference'
    )
    
    
def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / args.gif_name

    pipe, _ = load_models(device, args.model_dir, args.meta_dir, args.max_motion_length)
    print('args prompt: ', args.prompt)

    run_inference(
        pipe=pipe,
        prompt=args.prompt,
        num_steps=args.num_steps,
        max_motion_len = args.max_motion_length,
        seed=args.seed,
        device=device,
        outputs_path=args.output_dir
    )


if __name__ == "__main__":
    main()
