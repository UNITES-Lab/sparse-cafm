import os
import torch
import time
import json
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from networks import Generator, Discriminator
from torch import autograd
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle


class Config:
    """Config class"""

    def __init__(self, tag, root=""):
        self.tag = tag
        self.cli = False
        # self.wandb = True
        self.path = os.path.join(root, f"runs/{self.tag}")
        self.cm = "gray"
        self.data_path = ""
        self.mask_coords = []
        self.net_type = "conv-resize"
        self.image_type = "n-phase"
        self.l = 80
        self.n_phases = 2
        # Training hyperparams
        self.batch_size = 4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_iters = 400e3
        self.timeout = 1e12
        self.lrg = 0.0005
        self.lr = 0.0005
        self.Lambda = 10
        self.critic_iters = 10
        self.pw_coeff = 1
        self.ngpu = torch.cuda.device_count()
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = "cpu"
        self.conv_resize = True
        self.nz = 100
        # Architecture
        self.lays = 4
        self.laysd = 5
        # kernel sizes
        self.dk, self.gk = [4] * self.laysd, [4] * self.lays
        self.ds, self.gs = [2] * self.laysd, [2] * self.lays
        self.df, self.gf = [self.n_phases, 64, 128, 256, 512, 1], [
            self.nz,
            512,
            256,
            128,
            self.n_phases,
        ]
        self.dp, self.gp = [1] * self.laysd, [2] * self.lays
        # Last two layers conv resize (3,1,0)
        self.gk[-2:], self.gs[-2:], self.gp[-2:] = [3, 3], [1, 1], [0, 0]

    def update_params(self):
        self.df[0] = self.n_phases
        self.gf[-1] = self.n_phases

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f"{self.path}/config.json", "w") as f:
            json.dump(j, f)

    def load(self):
        with open(f"{self.path}/config.json", "r") as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_net_params(self):
        return self.dk, self.ds, self.df, self.dp, self.gk, self.gs, self.gf, self.gp

    def get_train_params(self):
        return (
            self.l,
            self.batch_size,
            self.beta1,
            self.beta2,
            self.lrg,
            self.lr,
            self.Lambda,
            self.critic_iters,
            self.nz,
        )


# check for existing models and folders
def check_existence(tag, root):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    check_D = os.path.exists(f"{root}/runs/{tag}/Disc.pt")
    check_G = os.path.exists(f"{root}/runs/{tag}/Gen.pt")
    if check_G or check_D:
        print(f"Models already exist for tag {tag}.")
        x = input(
            "To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n"
        )
        if x == "o":
            print("Overwriting")
            return True
        if x == "l":
            print("Loading previous model")
            return False
        elif x == "c":
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True


# set-up util
def initialise_folders(tag, overwrite, root):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f"{root}/runs")
        except:
            pass
        try:
            os.mkdir(f"{root}/runs/{tag}")
        except:
            pass


# training util
def preprocess(data_path, imtype, load=True):
    """[summary]

    :param imgs: [description]
    :type imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    # img = tifffile.imread(data_path)
    img = plt.imread(data_path)
    if imtype == "colour":
        img = img[:, :, :3]
        img = torch.tensor(img)
        if torch.max(img) > 1:
            img = img / torch.max(img)
        return img.permute(2, 0, 1), 3
    else:
        if len(img.shape) > 2:
            img = img[..., 0]
        if imtype == "n-phase":
            phases = np.unique(img)
            if len(phases) > 10:
                raise AssertionError("Image not one hot encoded.")
            x, y = img.shape
            img_oh = torch.zeros(len(phases), x, y)
            for i, ph in enumerate(phases):
                img_oh[i][img == ph] = 1
            return img_oh, len(phases)
        elif imtype == "grayscale":
            img = np.expand_dims(img, 0)
            img = torch.tensor(img)
            if torch.max(img) > 1:
                img = img / torch.max(img)
            return img, 1


def calculate_size_from_seed(seed, c):
    imsize = seed
    count = 0
    no_layers = len(c.gk)
    for k, s, p in zip(c.gk, c.gs, c.gp):
        if count < no_layers - 2:
            imsize = (imsize - 1) * s - 2 * p + k
        elif count == no_layers - 2:
            imsize = ((imsize - k + 2 * p) / s + 1).to(int)
            imsize = imsize * 2 + 2
        else:
            imsize = ((imsize - k + 2 * p) / s + 1).to(int)
        count += 1
    return imsize


def calculate_seed_from_size(imsize, c):
    count = 0
    no_layers = len(c.gk)
    for k, s, p in zip(c.gk, c.gs, c.gp):

        if count < no_layers - 2:
            imsize = ((imsize - k + 2 * p) / s + 1).to(int)
        elif count == no_layers - 2:
            imsize = (imsize - 1) * s - 2 * p + k
            imsize = ((imsize - 2) / 2).to(int)
        else:
            imsize = (imsize - 1) * s - 2 * p + k
        count += 1
    return imsize


def make_mask(training_imgs, c):

    y1, y2, x1, x2 = c.mask_coords
    ydiff, xdiff = y2 - y1, x2 - x1

    # seed for size of inpainting region
    seed = calculate_seed_from_size(torch.tensor([xdiff, ydiff]).to(int), c)
    # add 2 seed to each side to make the MSE region, the total G region
    img_seed = seed + 4
    G_out_size = calculate_size_from_seed(img_seed, c)
    mask_size = calculate_size_from_seed(seed, c)
    # THIS IS WHERE WE TELL D WHAT SIZE TO BE
    D_seed = img_seed
    x2, y2 = x1 + mask_size[0].item(), y1 + mask_size[1].item()
    xmid, ymid = (x2 + x1) // 2, (y2 + y1) // 2
    x1_bound, x2_bound, y1_bound, y2_bound = (
        xmid - G_out_size[0].item() // 2,
        xmid + G_out_size[0].item() // 2,
        ymid - G_out_size[1].item() // 2,
        ymid + G_out_size[1].item() // 2,
    )
    unmasked = training_imgs[:, x1_bound:x2_bound, y1_bound:y2_bound].clone()
    training_imgs[:, x1:x2, y1:y2] = 0
    mask = training_imgs[:, x1_bound:x2_bound, y1_bound:y2_bound]
    mask_layer = torch.zeros_like(training_imgs[0]).unsqueeze(0)
    unmasked = torch.cat([unmasked, torch.zeros_like(unmasked[0]).unsqueeze(0)])
    mask_layer[:, x1:x2, y1:y2] = 1
    mask = torch.cat((mask, mask_layer[:, x1_bound:x2_bound, y1_bound:y2_bound]))

    # save coords to c
    c.img_seed_x, c.img_seed_y = (img_seed[0].item(), img_seed[1].item())
    c.mask_coords = (x1, x2, y1, y2)
    c.G_out_size = (G_out_size[0].item(), G_out_size[1].item())
    c.mask_size = (mask_size[0].item(), mask_size[1].item())
    c.D_seed_x = D_seed[0].item()
    c.D_seed_y = D_seed[1].item()
    return mask, unmasked, G_out_size, img_seed, c


def update_pixmap_rect(raw, img, c, save_path=None, border=False):
    updated_pixmap = raw.clone().unsqueeze(0)
    x1, x2, y1, y2 = c.mask_coords
    lx, ly = c.mask_size
    x_1, x_2, y_1, y_2 = (
        (img.shape[2] - lx) // 2,
        (img.shape[2] + lx) // 2,
        (img.shape[3] - ly) // 2,
        (img.shape[3] + ly) // 2,
    )
    updated_pixmap[:, :, x1:x2, y1:y2] = img[:, :, x_1:x_2, y_1:y_2]
    updated_pixmap = post_process(updated_pixmap, c).permute(0, 2, 3, 1)
    if c.image_type == "grayscale":
        pm = updated_pixmap[0, ...]
    else:
        pm = updated_pixmap[0].numpy()
    if save_path:
        fig, ax = plt.subplots()
        if c.image_type == "grayscale":
            ax.imshow(pm, cmap="gray")
            rect_col = "#CC2825"
        else:
            ax.imshow(pm)
            rect_col = "#CC2825"
            # rect_col = 'white'

        if border:
            rect = Rectangle(
                (y1, x1),
                ly,
                lx,
                linewidth=1,
                ls="--",
                edgecolor=rect_col,
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig("data/temp/temp_fig.png", transparent=True, pad_inches=0)
        plt.close()
        if c.image_type == "grayscale":
            plt.imsave(c.temp_path, np.concatenate([pm for i in range(3)], -1))
        else:
            plt.imsave(c.temp_path, pm)
        return fig
    else:
        if c.image_type == "grayscale":
            pm = np.concatenate([pm for i in range(3)], -1)
        plt.imsave(c.temp_path, pm)
        return pm


def calc_gradient_penalty(
    netD: Discriminator,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    batch_size: int,
    lx: int,
    ly: int,
    device,
    gp_lambda: float,
    nc: int,
):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(
        batch_size, int(real_data.nelement() / batch_size)
    ).contiguous()
    alpha = alpha.view(batch_size, nc, lx, ly)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def batch_real_poly(img, l, bs, real_seeds):
    n_ph, _, _ = img.shape
    max_idx = len(real_seeds[0])
    idxs = torch.randint(max_idx, (bs,))
    data = torch.zeros((bs, n_ph, l, l))
    for i, idx in enumerate(idxs):
        x, y = real_seeds[0][idx], real_seeds[1][idx]
        data[i] = img[:, x : x + l, y : y + l]
    return data


def batch_real(img, lx, ly, bs, mask_coords):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    x1, x2, y1, y2 = mask_coords
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, lx, ly))
    for i in range(bs):
        x, y = torch.randint(x_max - lx, (1,)), torch.randint(y_max - ly, (1,))
        while (x1 < x + lx and x1 > x - lx) and (y1 < y + ly and y1 > y - ly):
            x, y = torch.randint(x_max - lx, (1,)), torch.randint(y_max - ly, (1,))
        data[i] = img[:, x : x + lx, y : y + ly]
    return data


def pixel_wise_loss(fake_img, real_img, unmasked, mode="mse", device=None):
    mask = real_img.clone().permute(1, 2, 0)
    mask = (mask[..., -1] == 0).unsqueeze(0)
    number_valid_pixels = mask.sum()
    mask = mask.repeat(fake_img.shape[0], fake_img.shape[1], 1, 1)
    fake_img = torch.where(mask == True, fake_img, torch.tensor(0).float().to(device))
    real_img = real_img.unsqueeze(0).repeat(fake_img.shape[0], 1, 1, 1)[:, 0:-1]
    real_img = torch.where(mask == True, real_img, torch.tensor(0).float().to(device))

    if mode == "mse":
        loss = torch.nn.MSELoss(reduction="sum")(fake_img, real_img) / (
            number_valid_pixels * fake_img.shape[0] * fake_img.shape[1]
        )
    elif mode == "ce":
        loss = -(
            real_img * torch.log(fake_img) + (1 - real_img) * torch.log(1 - fake_img)
        ).nanmean()

    return loss


# Evaluation util
def post_process(img: torch.Tensor, c: Config):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    if c.image_type == "n-phase":
        phases = np.arange(c.n_phases)
        color = iter(cm.get_cmap(c.cm)(np.linspace(0, 1, c.n_phases)))
        # color = iter([[0,0,0],[0.5,0.5,0.5], [1,1,1]])
        img = torch.argmax(img, dim=1)
        if len(phases) > 10:
            raise AssertionError("Image not one hot encoded.")
        bs, x, y = img.shape
        out = torch.zeros((bs, 3, x, y))
        for b in range(bs):
            for i, ph in enumerate(phases):
                col = next(color)
                col = torch.tile(
                    torch.Tensor(col[0:3]).unsqueeze(1).unsqueeze(1), (x, y)
                )
                out[b] = torch.where((img[b] == ph), col, out[b])
        out = out
    else:
        out = img
    return out


def crop(fake_data, l, miniD=False, l_mini=16, offset=8):
    w = fake_data.shape[2]
    h = fake_data.shape[3]
    x1, x2 = (w - l) // 2, (w + l) // 2
    y1, y2 = (h - l) // 2, (h + l) // 2

    out = fake_data[:, :, x1:x2, y1:y2]
    return out


def init_noise(batch_size: int, nz, c: Config, device) -> torch.Tensor:
    """
    Create and return noise tensor.
    TODO: what is the shape?
    """
    
    noise = torch.randn(1, nz, c.seed_x, c.seed_y, device=device)
    noise = torch.tile(noise, (batch_size, 1, 1, 1))
    noise.requires_grad = True
    return noise


def make_noise(noise, device, mask_noise=False, delta=[1, 1]):
    # zeros in mask are fixed, ones are random
    mask = torch.zeros_like(noise).to(device)
    _, _, x, y = mask.shape
    if mask_noise:
        dx = torch.div(delta[0], 2, rounding_mode="floor")
        dy = torch.div(delta[1], 2, rounding_mode="floor")
        if dx > 0 and dy > 0:
            mask[:, :, x // 2 - dx : x // 2 + dx, y // 2 - dy : y // 2 + dy] = 1
        elif dx == 0:
            mask[:, :, x // 2, y // 2 - dy : y // 2 + dy] = 1
        elif dy == 0:
            mask[:, :, x // 2 - dx : x // 2 + dx, y // 2] = 1
        rand = torch.randn_like(noise).to(device) * mask
        noise = noise * (mask == 0) + rand
    else:
        noise = torch.randn_like(noise).to(device)
    return noise


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class RectWorker:
    """
    Code: https://github.com/tldr-group/microstructure-inpainter
    Paper: https://arxiv.org/pdf/2210.06997
    """

    def __init__(
        self,
        c: Config,
        netG: Generator,
        netD: Discriminator,
        training_imgs: torch.Tensor,
        nc: int,
        mask: Optional[torch.Tensor]=None,
        unmasked=None,
    ):
        super().__init__()
        self.c: Config = c
        self.netG: Generator = netG
        self.netD: Discriminator = netD
        self.training_imgs: torch.Tensor = training_imgs
        self.nc: int = nc
        self.mask: torch.Tensor = mask
        self.unmasked: torch.Tensor = unmasked
        self.quit_flag = False
        self.opt_whilst_train = True

        # self.opt_whilst_train = not c.cli

    def stop(self):
        self.quit_flag = True

    def train(self, wandb=None):
        """[summary]

        :param c: [description]
        :type c: [type]
        :param Gen: [description]
        :type Gen: [type]
        :param Disc: [description]
        :type Disc: [type]
        :param offline: [description], defaults to True
        :type offline: bool, optional
        """

        # Assign torch device
        # NOTE: really bad code...
        overwrite = True
        c: Config = self.c
        Gen: Generator = self.netG
        Disc: Discriminator = self.netD
        training_imgs: torch.Tensor = self.training_imgs
        nc: int = self.nc
        mask: torch.Tensor = self.mask
        unmasked = self.unmasked
        ngpu = c.ngpu
        tag = c.tag
        path = c.path
        device = torch.device(
            c.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        print(f"Using {ngpu} GPUs")
        print(device, " will be used.\n")
        print(
            f"Data shape: {training_imgs.shape}. Inpainting shape: {c.mask_size} Seed size: {c.img_seed_x, c.img_seed_y}"
        )

        cudnn.benchmark = True

        # train parameters
        (
            l,
            batch_size,
            beta1,
            beta2,
            lrg,
            lr,
            Lambda,
            critic_iters,
            nz,
        ) = c.get_train_params()

        mask = mask.to(device)
        unmasked = unmasked.to(device)

        # init noise
        noise = init_noise(1, nz, c, device)

        # TODO: we pass in fns; should just be model objects
        netG = Gen.to(device)
        netD = Disc.to(device)

        # NOTE: we remove this wonky support for multiple GPUs
        # -------------------------------------------------------
        # if ("cuda" in str(device)) and (ngpu > 1):
        #     Dnet = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        #     netG = nn.DataParallel(netG, list(range(ngpu))).to(device)

        # optimizer for discriminator/generator
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

        # NOTE: here we load model + noise from memory; I think we can just disable this
        # -------------------------------------------------------
        # if not overwrite:
        #     netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
        #     netD.load_state_dict(torch.load(f"{path}/Disc.pt"))
        #     noise = torch.load(f"{c.path}/noise.pt")

        # NOTE: disable wandb logging
        # if c.wandb:
        #     wandb.wandb_init(tag, netG, netD, offline=False)

        i = 0
        t = 0

        # start timing training
        if ("cuda" in str(device)) and (ngpu > 1):
            start_overall = torch.cuda.Event(enable_timing=True)
            end_overall = torch.cuda.Event(enable_timing=True)
            start_overall.record()
        else:
            start_overall = time.time()

        while not self.quit_flag and t < c.timeout and i < c.max_iters:

            # Discriminator Training
            netD.zero_grad()
            netG.train()

            d_noise = torch.randn_like(noise).to(device)
            fake_data = netG(d_noise).detach()

            # fake_data = crop(fake_data,dl)
            real_data = batch_real(
                training_imgs,
                fake_data.shape[-2],
                fake_data.shape[-1],
                batch_size,
                c.mask_coords,
            ).to(device)

            # Train on real
            out_real = netD(real_data).mean()

            # train on fake images
            out_fake = netD(fake_data).mean()

            gradient_penalty = calc_gradient_penalty(
                netD,
                real_data,
                fake_data,
                batch_size,
                fake_data.shape[-2],
                fake_data.shape[-1],
                device,
                Lambda,
                nc,
            )

            # Compute the discriminator loss and backprop
            wass = out_fake - out_real
            disc_cost = wass + gradient_penalty
            disc_cost.backward()

            # take optimization step on discriminator
            optD.step()

            # if c.wandb:
            #     wandb.log(
            #         {"D_real": out_real.item(), "D_fake": out_fake.item()}, step=i
            #     )

            # Generator training
            if (i % int(critic_iters)) == 0:
                netG.zero_grad()
                noise_G = torch.randn_like(noise).to(device)

                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                output = -netD(fake_data).mean()

                noise_G = make_noise(noise, device, mask_noise=True, delta=[-1, -1])
                fake_data = netG(noise_G)
                pw = pixel_wise_loss(
                    fake_data, mask, unmasked, mode="mse", device=device
                )
                output += pw * c.pw_coeff

                # Calculate loss for G and backprop
                output.backward(retain_graph=True)
                optG.step()

            # Every 100 iters log images and useful metrics
            if i % 100 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), f"{path}/Gen.pt")
                    torch.save(netD.state_dict(), f"{path}/Disc.pt")
                    torch.save(noise, f"{path}/noise.pt")

                    if ("cuda" in str(device)) and (ngpu > 1):
                        end_overall.record()
                        torch.cuda.synchronize()
                        t = start_overall.elapsed_time(end_overall)
                    else:
                        end_overall = time.time()
                        t = end_overall - start_overall
                    if self.opt_whilst_train:
                        plot_noise = make_noise(
                            noise.detach().clone(),
                            device,
                            mask_noise=True,
                            delta=[-1, -1],
                        )
                        img = netG(plot_noise).detach()
                        pixmap = update_pixmap_rect(training_imgs, img, c)

                        if c.cli:
                            print(
                                f"Iter: {i}, Time: {t:.1f}, MSE: {pw.sum().item():.2g}, Wass: {abs(wass.item()):.2g}"
                            )
                            if c.wandb:
                                wandb.log(
                                    {
                                        "mse": pw.nanmean().item(),
                                        "wass": wass.item(),
                                        "gp": gradient_penalty.item(),
                                        "raw out": wandb.Image(img[0].cpu()),
                                        "inpaint out": wandb.Image(pixmap),
                                    },
                                    step=i,
                                )
                        else:
                            self.progress.emit(i, t, pw.item(), abs(wass.item()))
                    else:
                        print(f"Iter: {i}, Time {t:.1f}")

            i += 1

            if i == c.max_iters:
                print(f"Max iterations reached: {i}")

            if self.quit_flag:
                self.finished.emit()
                print("Quitting training")

        if t > c.timeout:
            print(f"Timeout: {t:.2g}")

        self.finished.emit()

        print("TRAINING FINISHED")

    def generate(self, save_path=None, border=False, delta=None):

        if self.verbose:
            print("Generating new inpainted image")

        device = torch.device(
            self.c.device_name
            if (torch.cuda.is_available() and self.c.ngpu > 0)
            else "cpu"
        )
        netG = self.netG().to(device)
        netD = self.netD().to(device)

        if ("cuda" in str(device)) and (self.c.ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(self.c.ngpu)))).to(device)
            netG = nn.DataParallel(netG, list(range(self.c.ngpu))).to(device)

        netG.load_state_dict(torch.load(f"{self.c.path}/Gen.pt"))
        netD.load_state_dict(torch.load(f"{self.c.path}/Disc.pt"))

        noise = torch.load(f"{self.c.path}/noise.pt")
        netG.eval()

        with torch.no_grad():
            # delta is an int that dictates how much of the centre of the seed is random
            if delta is None:
                if min(noise.shape[2:]) < 10:
                    mask_noise = False
                else:
                    delta = torch.tensor(noise.shape[2:]) - 10
                    mask_noise = True
            elif delta == "rand":
                mask_noise = False
            plot_noise = make_noise(
                noise.detach().clone(), device, mask_noise=mask_noise, delta=delta
            )
            img = netG(plot_noise).detach()
            f = update_pixmap_rect(
                self.training_imgs, img, self.c, save_path=save_path, border=border
            )
            if save_path:
                axs = f.axes
                f.savefig(f"{save_path}_border.png", transparent=True)
                for ax in axs:
                    ax.patches = []
                f.savefig(f"{save_path}.png", transparent=True)
            return img
