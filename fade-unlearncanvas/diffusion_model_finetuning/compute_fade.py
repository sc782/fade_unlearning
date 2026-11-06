"""
One direction of FADE for latent diffusion models: an upper bound on
KL(p(x_0 | c) || q(x_0 | c)), computed while p samples.

The driver model p generates images with the stochastic DDIM sampler (eta > 0,
`--steps` steps, classifier-free guidance `--scale`); at every step both p and the
other model q predict the guided noise at the current latent x_tau, and

    D(x) = (1/d) sum_tau  w_tau * || eps~_p(x_tau, tau) - eps~_q(x_tau, tau) ||^2,
    eps~ = eps(x, tau, "") + scale * (eps(x, tau, c) - eps(x, tau, ""))

is accumulated, where d = C x H x W is the latent dimension. One trajectory of p is
scored against any number of other models q. Each sampler step is Gaussian with the
same variance sigma_tau^2 under p and q, and its mean is linear in eps~ with
coefficient c_tau, so d * (summand) is the exact KL(p(x_prev | x_tau) || q(x_prev | x_tau))
with w_tau = c_tau^2 / (2 sigma_tau^2). By the chain rule and data processing,
d * E_p[D] >= KL(p(x_0 | c) || q(x_0 | c)) for the latents the sampler outputs (and the
images they decode to): D is the bound in nats per latent dimension, which keeps
values comparable across image resolutions.

Usage
-----
python compute_fade.py --theme Monet --driver retain0=logs/retain_seed0.ckpt \
    --others retain1=logs/retain_seed1.ckpt full=../ckpts/diffusion/compvis/style50/compvis.ckpt

writes output_fade/<theme>/<driver>_vs_<other>.pt (one direction: images from the
driver), with a per-image `diffs` tensor in nats per latent dimension, and the driver's
images to output_images/<driver>/<theme>/. aggregate.py sums the two directions.
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from constants.const import class_available
from ldm.modules.attention import CrossAttention
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from ldm.util import instantiate_from_config


def _sdpa_forward(self, x, context=None, mask=None):
    """CrossAttention.forward via fused scaled-dot-product attention (same math)."""
    assert mask is None
    context = x if context is None else context
    q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
    q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return self.to_out(out.transpose(1, 2).flatten(2))


if hasattr(F, "scaled_dot_product_attention"):  # torch >= 2.0
    CrossAttention.forward = _sdpa_forward


def named_path(arg):
    name, path = arg.split("=", 1)
    return name, path


def load_model(config, ckpt, device):
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt, map_location="cpu")
    missing, _ = model.load_state_dict(state["state_dict"], strict=False)
    if missing:
        raise ValueError(f"{ckpt} is missing {len(missing)} weights, e.g. {missing[:3]}")
    return model.to(device).eval()


def sampler_schedule(model, steps, eta):
    """DDIM timesteps (ascending), sigma_tau, and the coefficients of the step mean
    mean = a_tau * x_tau + c_tau * eps, so the per-step KL weight is c^2 / (2 sigma^2)."""
    alphacums = model.alphas_cumprod.cpu().numpy()
    timesteps = make_ddim_timesteps("uniform", steps, model.num_timesteps, verbose=False)
    sigmas, alphas, alphas_prev = make_ddim_sampling_parameters(alphacums, timesteps, eta, verbose=False)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    a_x = np.sqrt(alphas_prev / alphas)
    c_eps = np.sqrt(1 - alphas_prev - sigmas ** 2) - np.sqrt(alphas_prev) * np.sqrt(1 - alphas) / np.sqrt(alphas)
    weights = c_eps ** 2 / (2 * sigmas ** 2)
    return timesteps, sigmas, a_x, c_eps, weights


def guided_eps(model, x, t, cond, uncond, scale):
    e_u, e_c = model.apply_model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([uncond, cond])).chunk(2)
    return e_u + scale * (e_c - e_u)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate_sd.yaml")
    parser.add_argument("--driver", required=True, type=named_path,
                        help="name=checkpoint whose sampling chain is followed")
    parser.add_argument("--others", required=True, nargs="+", type=named_path,
                        help="name=checkpoint models compared against the driver")
    parser.add_argument("--theme", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100)))
    parser.add_argument("--classes", nargs="+", default=class_available)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=9.0)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16",
                        help="UNet precision (fp16 = autocast, as in the image sampler)")
    parser.add_argument("--out_dir", default="output_fade")
    parser.add_argument("--image_dir", default="output_images",
                        help="where to save the driver's images ('' to skip)")
    args = parser.parse_args()
    if args.eta <= 0:
        parser.error("the bound needs a stochastic sampler (eta > 0)")

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.load(args.config)
    driver_name, driver = args.driver[0], load_model(config, args.driver[1], device)
    others = {name: load_model(config, path, device) for name, path in args.others}
    timesteps, sigmas, a_x, c_eps, weights = sampler_schedule(driver, args.steps, args.eta)

    prompts = [f"A {c} image in {args.theme.replace('_', ' ')} style." for c in args.classes]
    shape = (len(prompts), 4, args.H // 8, args.W // 8)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=args.dtype == "fp16")
    with autocast:
        conds = {m: (m.get_learned_conditioning(prompts), m.get_learned_conditioning([""] * len(prompts)))
                 for m in [driver, *others.values()]}
    image_dir = args.image_dir and os.path.join(args.image_dir, driver_name, args.theme)
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    per_step, names = {name: [] for name in others}, []
    for seed in tqdm(args.seeds):
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=gen, device=device)
        contrib = {name: torch.zeros(len(prompts), len(timesteps), dtype=torch.float64) for name in others}
        for i in reversed(range(len(timesteps))):
            t = torch.full((len(prompts),), int(timesteps[i]), device=device, dtype=torch.long)
            with autocast:
                e_p = guided_eps(driver, x, t, *conds[driver], args.scale).float()
                for name, other in others.items():
                    e_q = guided_eps(other, x, t, *conds[other], args.scale).float()
                    contrib[name][:, i] = weights[i] * (e_p - e_q).double().pow(2).mean(dim=(1, 2, 3)).cpu()
            noise = torch.randn(shape, generator=gen, device=device)
            x = a_x[i] * x + c_eps[i] * e_p + sigmas[i] * noise
        for name in others:
            per_step[name].append(contrib[name])
        names += [f"{args.theme}_{c}_seed{seed}" for c in args.classes]
        if image_dir:
            with autocast:
                imgs = driver.decode_first_stage(x)
            imgs = ((imgs.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for img, name in zip(imgs, names[-len(prompts):]):
                Image.fromarray(img).save(os.path.join(image_dir, f"{name}.jpg"))

    out_dir = os.path.join(args.out_dir, args.theme)
    os.makedirs(out_dir, exist_ok=True)
    for name in others:
        steps_q = torch.cat(per_step[name])
        out = os.path.join(out_dir, f"{driver_name}_vs_{name}.pt")
        torch.save({"diffs": steps_q.sum(1), "per_step": steps_q, "names": names,
                    "driver": args.driver, "other": dict(args.others)[name],
                    "timesteps": torch.as_tensor(timesteps), "weights": torch.as_tensor(weights),
                    "latent_dim": int(np.prod(shape[1:])), "args": vars(args)}, out)
        print(f"{args.theme} {driver_name} vs {name}: E[D] = {steps_q.sum(1).mean():.3f} "
              f"over {len(names)} images -> {out}")


if __name__ == "__main__":
    main()
