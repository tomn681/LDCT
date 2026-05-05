<<<<<<< HEAD
# train_flow.py  (versión flow-matching, sin importar Unet2DCond salvo que se pida)
=======
>>>>>>> 04d2298b51031f866abece0536f625654bd72427
import os
import torch
import torch.nn.functional as F

from pathlib import Path
from config import config
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate import notebook_launcher
<<<<<<< HEAD

from models.DiffUNet2D import model as Unet2D  # UNet base (sin atención)

# >>>>>> IMPORTANTE: NO importamos Unet2DCond aquí para evitar que se instancie al importar
# from models.DiffUNet2DCond import model as Unet2DCond

from diffusers.utils import make_image_grid
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms

from utils.dataset import DefaultDataset, CombinationDataset  # si lo usas luego
from diffusers import FlowMatchEulerDiscreteScheduler


# --------------------------
# Selección de modelo/dataset
# --------------------------
# UNet sin atención
unet = Unet2D

# Dataset
if config.conditioning is not None:
    dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, diff=False)
    test_dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, diff=False, train=False)
else:
    dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices)

# Si alguien pone 'dual' por error, lo bloqueamos aquí (o haz lazy import si lo necesitas)
if config.conditioning == 'dual':
    raise NotImplementedError(
        "conditioning='dual' no está implementado en esta versión flow-matching. "
        "Usa conditioning='concatenate' o None."
    )
    # Si quisieras habilitarlo en el futuro:
    # from models.DiffUNet2DCond import model as Unet2DCond
    # unet = Unet2DCond

# DataLoader
loader_args = dict(batch_size=config.train_batch_size, num_workers=4, pin_memory=True, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(dataset, **loader_args)

# Scheduler (solo para timesteps e inferencia)
noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=config.num_train_timesteps)

# Optimizador y LR
optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# --------------------------
# Utils Hub
# --------------------------
=======
from models.DiffUNet2D import model as Unet2D
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import make_image_grid
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.dataset import DefaultDataset
from utils.sampler import SamplingPipeline


# Flow-matching training entrypoint built to mirror train.py without modifying it.

>>>>>>> 04d2298b51031f866abece0536f625654bd72427
def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


<<<<<<< HEAD
# --------------------------
# Sampling pipeline (flow-matching)
# --------------------------
class FlowMatchSamplingPipeline(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, num_train_timesteps: int, conditioning: str = None):
        super().__init__()
        self.unet = unet
        self.num_train_timesteps = num_train_timesteps
        self.conditioning = conditioning

    @torch.no_grad()
    def __call__(self, batch_size: int, images=None, num_inference_steps: int = None, height=None, width=None, device=None):
        device = device or next(self.unet.parameters()).device
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.num_train_timesteps)
        if num_inference_steps is None:
            num_inference_steps = getattr(config, "num_inference_steps", 50)
        scheduler.set_timesteps(num_inference_steps, device=device)

        C = images["target"].shape[1] if (images is not None and isinstance(images, dict) and "target" in images) else 1
        H = height or config.image_size
        W = width or config.image_size

        x = torch.randn(batch_size, C, H, W, device=device)
        cond = None
        if self.conditioning == "concatenate":
            cond = images["image"].to(device)

        for t in scheduler.timesteps:
            model_in = x if self.conditioning != "concatenate" else torch.cat([x, cond], dim=1)
            v = self.unet(model_in, t, return_dict=False)[0]        # predicción de velocidad
            x = scheduler.step(model_output=v, timestep=t, sample=x).prev_sample

        imgs = (x.clamp(-1, 1) + 1) / 2
        imgs = (imgs * 255).round().to(torch.uint8).cpu()
        pil_list = [transforms.ToPILImage()(img) for img in imgs]
        return type("Obj", (), {"images": pil_list})


# --------------------------
# Entrenamiento (flow-matching)
# --------------------------
def train_loop(cfg, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
=======
def build_dataloaders(cfg):
    if cfg.conditioning is not None:
        dataset = DefaultDataset("./DefaultDataset", img_size=cfg.image_size, s_cnt=cfg.slices, diff=False)
    else:
        dataset = DefaultDataset("./DefaultDataset", img_size=cfg.image_size, s_cnt=cfg.slices)

    loader_args = dict(batch_size=cfg.train_batch_size, num_workers=4, pin_memory=True)
    return torch.utils.data.DataLoader(dataset, shuffle=True, **loader_args)


def train_loop(cfg, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
>>>>>>> 04d2298b51031f866abece0536f625654bd72427
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(cfg.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if cfg.push_to_hub:
            repo_name = get_full_repo_name(Path(cfg.output_dir).name)
            repo = Repository(cfg.output_dir, clone_from=repo_name)
        elif cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
<<<<<<< HEAD
        accelerator.init_trackers("train_flow_matching")

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
=======
        accelerator.init_trackers("train_flowmatch")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
>>>>>>> 04d2298b51031f866abece0536f625654bd72427
    )

    global_step = 0

    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
<<<<<<< HEAD
            x0 = batch["target"].to(accelerator.device)
            z  = torch.randn_like(x0)
            bs = x0.shape[0]

            # t discreto para el UNet (drop-in con tu embedding de DDPM)
            t_idx  = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=x0.device).long()
            t_cont = (t_idx.float() + 0.5) / noise_scheduler.config.num_train_timesteps
            t      = t_cont.view(bs, 1, 1, 1)

            # Trayectoria determinista
            xt = (1.0 - t) * x0 + t * z

            # Cond por concatenación
            if cfg.conditioning == "concatenate":
                cond = batch["image"].to(accelerator.device)
                model_in = torch.cat((xt, cond), dim=1)
            elif cfg.conditioning is None:
                model_in = xt
            else:
                raise NotImplementedError(f"conditioning='{cfg.conditioning}' no soportado.")

            # Etiqueta: velocidad
            v_target = z - x0

            with accelerator.accumulate(unet):
                v_pred = unet(model_in, t_idx, return_dict=False)[0]
                loss = F.mse_loss(v_pred, v_target)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
=======
            clean_images = batch["target"]
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample continuous time for flow matching in [0, 1].
            t = torch.rand(bs, device=clean_images.device)
            timesteps = (t * (noise_scheduler.config.num_train_timesteps - 1)).long()

            # Linear path from data to noise.
            x_t = (1.0 - t[:, None, None, None]) * clean_images + t[:, None, None, None] * noise

            if cfg.conditioning == "concatenate":
                model_input = torch.cat((x_t, batch["image"]), dim=1)
            else:
                model_input = x_t

            with accelerator.accumulate(model):
                # Predict velocity (noise - data) with rectified flow objective.
                model_pred = model(model_input, timesteps, return_dict=False)[0]
                target = noise - clean_images
                loss = F.mse_loss(model_pred, target)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
>>>>>>> 04d2298b51031f866abece0536f625654bd72427

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
<<<<<<< HEAD
            pipeline = FlowMatchSamplingPipeline(
                unet=accelerator.unwrap_model(unet),
                num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                conditioning=cfg.conditioning,
            )
            if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
                evaluate(cfg, epoch, pipeline, batch)

=======
            pipeline = SamplingPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, conditioning=cfg.conditioning
            )
            if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
                imgz = batch if cfg.conditioning is not None else None
                evaluate(cfg, epoch, pipeline, imgz)
>>>>>>> 04d2298b51031f866abece0536f625654bd72427
            if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
                if cfg.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
<<<<<<< HEAD
                    save_dir = cfg.output_dir
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(accelerator.unwrap_model(unet).state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                    with open(os.path.join(save_dir, "flow_config.txt"), "w") as f:
                        f.write(f"num_train_timesteps={noise_scheduler.config.num_train_timesteps}\n")
                        f.write(f"conditioning={cfg.conditioning}\n")


def evaluate(cfg, epoch, pipeline, batch):
    images = pipeline(
        batch_size=cfg.eval_batch_size,
        images=batch if cfg.conditioning is not None else None,
        num_inference_steps=getattr(cfg, "num_inference_steps", 50),
        device=next(pipeline.unet.parameters()).device,
    ).images[:cfg.eval_batch_size]

    rows = max(1, cfg.eval_batch_size // 4)
    cols = min(4, cfg.eval_batch_size)
    image_grid = make_image_grid(images, rows=rows, cols=cols)
=======
                    pipeline.save_pretrained(cfg.output_dir)


def evaluate(cfg, epoch, pipeline, images):
    images = pipeline(batch_size=cfg.eval_batch_size, images=images).images[: cfg.eval_batch_size]
    image_grid = make_image_grid(images, rows=cfg.eval_batch_size // 4, cols=4)
>>>>>>> 04d2298b51031f866abece0536f625654bd72427

    test_dir = os.path.join(cfg.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


<<<<<<< HEAD
preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


if __name__ == '__main__':
    args = (config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)

=======
if __name__ == "__main__":
    # Adjust config fields at runtime to avoid collisions with DDPM runs.
    config.model_name = f"{config.model_name}_FlowMatch"
    config.output_dir = f"train/{config.model_name.lower()}-{config.mixed_precision}-{config.image_size}-{config.slices}-{config.seed}"

    model = Unet2D
    train_dataloader = build_dataloaders(config)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=config.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)
>>>>>>> 04d2298b51031f866abece0536f625654bd72427
