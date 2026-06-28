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
    def __init__(self, dit, text_embedder, decoder, device, meta_dir,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2):

        self.dit = dit
        self.dit.eval()
        self.text_embedder = text_embedder
        self.decoder = decoder
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.mean = np.load(pjoin(meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(meta_dir, 'std.npy'))

        with open(pjoin(meta_dir, 'part_mapping.json'), 'r') as f:
            mapping = json.load(f)

        self.part_names = mapping['part_names']
        self.d_part_max = mapping['d_part_max']
        self.joints_num = mapping['joints_num']


    def denormalize_motion(self, motion):
        return motion * self.std + self.mean
    
    @torch.no_grad()
    def __calltest__(self, prompt, generator, num_inference_steps=50, seed=42, latent_shape=(1, 512), eta = 0.0):
        #generator = torch.Generator(device=self.device).manual_seed(seed)

        z1 = torch.randn(latent_shape, generator=generator, device=self.device)
        z2 = z1.clone()

        #cond = self.text_embedder.encode(prompt, convert_to_tensor = True, device = str(self.device)).clone()
        text_tokens, text_mask = self.text_embedder.encode_tokens(["a person walking slowly"])
        text_tokens1, text_mask1 = self.text_embedder.encode_tokens(["a person twists from side to side"])
        print(prompt, num_inference_steps)
        print(text_tokens.mean().item(), text_tokens.std().item())
        print(text_tokens1.mean().item(), text_tokens1.std().item())

        for d_step in range(num_inference_steps):
            t = torch.full((z1.shape[0], ), fill_value = d_step / (num_inference_steps - 1), device = self.device)
            t = t.clamp(1e-4, 1.0)
            d = torch.zeros(z1.shape[0], device=self.device)
            alpha = 1.0 / num_inference_steps

            v1 = self.dit(z1, t, d, text_tokens, text_mask)
            z1 = z1 + alpha * v1

            block = self.dit.blocks[-1]
            cross_out_1 = block.last_cross_out # shape [B, L_motion, dim]

            v2 = self.dit(z2, t, d, text_tokens1, text_mask1)
            z2 = z2 + alpha * v2

            block = self.dit.blocks[-1]
            cross_out_2 = block.last_cross_out # shape [B, L_motion, dim]

            diff = (cross_out_1 - cross_out_2).abs().mean().item()
            print("Mean |cross_out(text1) - cross_out(text2)|:", diff)

            pred_diff = (v1 - v2).abs().mean().item()
            print("Mean |pred(text1) - pred(text2)|:", pred_diff)

        motion1 = self.decoder(z1)
        motion2 = self.decoder(z2)
        denormalized_motion1 = self.denormalize_motion(motion1[0])
        denormalized_motion2 = self.denormalize_motion(motion2[0])

        motion_joints1 = recover_from_ric(denormalized_motion1, self.joints_num)
        motion_joints2 = recover_from_ric(denormalized_motion2, self.joints_num)

        return motion_joints1, motion_joints2
    
    @torch.no_grad()
    def __call__(self, prompt, generator, num_inference_steps=50, seed=42, latent_shape=(1, 512), eta = 0.0):
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
    #text_embedder = get_pretrained_text_encoder(device)
    #text_embedder.eval()
    text_encoder = TextTokenEncoder(device = device).to(device)
    text_encoder.eval()
    dit_chkpt = torch.load(pjoin(model_dir, 'dit_crossattn_debug.tar'), map_location = device)
    dit = DiT(
        input_size = 512,
        hidden_size=1152,
        text_dim=384,
        max_seq_len=max_motion_length//4
    )
    dit.load_state_dict(dit_chkpt['dit'])
    dit.to(device)
    dit.eval()

    pipe = MotionPipeline(
        dit=dit,
        text_embedder=text_encoder,
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
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            saved_path = gif_path
        except Exception as exec:
            saved_path = None
            print('GIF file not saved: ', exec)
    plt.close(fig)
    return saved_path



@torch.no_grad()
def run_inference(pipe, prompt, num_steps, seed, device, outputs_path):

    # Replace this call with your actual pipeline invocation
    #prompt = 'run'
    gen = torch.Generator(device).manual_seed(42)
    result = pipe(
        prompt=prompt,
        generator = gen,
        num_inference_steps=num_steps
    )

    motion_joints = result
    all_gif_files = [fname for fname in os.listdir(outputs_path) if ".gif" in fname]
    file_id = len(all_gif_files)
    render_skeleton_animation(
        joints_recon=motion_joints,
        output_path_no_ext=pjoin(outputs_path, f'inference_test_clip_{file_id}'),
        clip_id=f'inference_test_clip_{file_id}',
        text = prompt,
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
        seed=args.seed,
        device=device,
        outputs_path=args.output_dir
    )


if __name__ == "__main__":
    main()
