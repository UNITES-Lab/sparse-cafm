import os
import rich
import torch
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from rich.console import Console
from rich.rule import Rule
from rich.pretty import Pretty
from pprint import pprint
from torch.utils.data import DataLoader

from src.util.logger import Logger
from src.util.metrics import PSNR, SSIM
from src.util.celano_lab_scripts import calculate_abs_diff_between_samples, pcnt_diff_surface_roughness, compute_surface_roughness, process_image
from src.models.our_method.swin_cafm import SwinCAFM
from src.models.prev_methods.gpr import GPR
from src.datasets.mos2_sr import MOS2SRDataset, MOS2_SEF_SRC_DIR, MOS2_SAPPHIRE_DIR, MOS2_SILICON_DIR

warnings.simplefilter("ignore")

NUM_TRIALS = 1028
console = Console()


@torch.no_grad()
def normalize(X: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    X_mean = X.mean()
    X_std = X.std()
    if X_std == 0:
        return X * 0 + mu
    return (X - X_mean) / X_std * sigma + mu


@torch.no_grad()
def eval(fp: str, formulation: str, dataset_name: str, upsampling_ratio: int) -> None:
    """
    Evaluate a super-resolution model for a given dataset/resolution/upsampling-ratio.
    """

    logger = Logger(args.exp_root, args.exp_name)

    # load model obj
    # model: SwinCAFM = torch.load(fp).cuda().float()
    # model.eval()
    
    # HACK: use GPR as basline
    model = GPR(sr=upsampling_ratio)
    
    src_dir = ""
    if dataset_name == "mos2-sef": src_dir = MOS2_SEF_SRC_DIR
    if dataset_name == "sapphire": src_dir = MOS2_SAPPHIRE_DIR
    if dataset_name == "silicon": src_dir = MOS2_SILICON_DIR

    dataset = MOS2SRDataset(
        src_dir=src_dir, 
        split="val",
        upsample_factor=upsampling_ratio, 
        steps_per_epoch=NUM_TRIALS
    )
    BATCH_SIZE  = 32
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    total_pred_sr = []
    total_baseline_sr = []

    total_baseline_errs = None
    total_pred_errs = None
    num_samples = 0

    psnr_vals = []
    ssim_vals = []

    # evaluate model
    for batch in tqdm(data_loader, desc=f"Processing SR: {upsampling_ratio} | dataset: {dataset_name}"):
        
        y       : torch.Tensor = batch[f"{formulation}"]
        y_sparse: torch.Tensor = batch[f"{formulation}_sparse"]
        y_unnorm: torch.Tensor = batch[f"{formulation}_unnorm"]

        y        = y.cuda().float()
        y_sparse = y_sparse.cuda().float()
        y_hat    = model(y_sparse).cuda().float()

        if formulation=="y":

            psnr = PSNR(y, y_hat).item()
            y_c = y.unsqueeze(1).repeat(1, 3, 1, 1)
            y_hat_c = y_hat.unsqueeze(1).repeat(1, 3, 1, 1)
            ssim = SSIM(y_c, y_hat_c).item()

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

            # --- NOTE: scale + shift ---
            # std normal -> original current/topo map mean/std
            # mu = dataset.current_maps_mean
            # sigma = dataset.current_maps_std
            
            # y        = normalize(y, mu, sigma)
            # y_hat    = normalize(y_hat, mu, sigma)
            # y_sparse = normalize(y_sparse, mu, sigma)

        # # iterate over all samples in the batch
        # for i in range(y_hat.size(0)):
            
        #     sample_y = y[i]
        #     sample_y_sparse = y_sparse[i]
        #     sample_y_hat = y_hat[i]

        #     if formulation=="X":

        #         # HACK: normalize sf
        #         # current = compute_surface_roughness(sample_y_hat) 
        #         # target = compute_surface_roughness(sample_y_sparse)
        #         # scale = target / current
        #         # sample_y_hat: torch.Tensor = (sample_y_hat.mean()) + \
        #         #     scale * (sample_y_hat - sample_y_hat.mean())
  
        #         total_baseline_sr.append(pcnt_diff_surface_roughness(sample_y, sample_y_sparse))
        #         total_pred_sr.append(pcnt_diff_surface_roughness(sample_y, sample_y_hat))
            
        #     elif formulation=="y":

        #         # NOTE: scale -> y_sparse mean
        #         alpha = sample_y_sparse.mean() / sample_y_hat.mean()
        #         sample_y_hat *= alpha

        #         # baseline
        #         current_baseline_errs = calculate_abs_diff_between_samples(sample_y, sample_y_sparse, 2.0)
                
        #         # experiment
        #         current_pred_errs = calculate_abs_diff_between_samples(sample_y, sample_y_hat, 2.0)
                
        #         # record results
        #         if total_pred_errs is None:
        #             total_pred_errs     = {"baseline_"  + key: value for key, value in current_pred_errs.items()}
        #             total_baseline_errs = {"predicted_" + key: value for key, value in current_baseline_errs.items()}
        #         else:
        #             for key in current_pred_errs:
        #                 total_pred_errs["baseline_"  + key]     += current_pred_errs[key]
        #             for key in current_baseline_errs:
        #                 total_baseline_errs["predicted_" + key] += current_baseline_errs[key]
        #     else:
        #         raise Exception()

        #     num_samples += 1
    
    if formulation=="X":
        
        console = Console()
        console.print(Rule(f"[bold blue]Results for SR: {upsampling_ratio} | dataset: {dataset_name}"))
        console.print("[bold magenta]Average %diffs (y, y_sparse):")
        console.print(Pretty(np.array(total_baseline_sr).mean(), indent_guides=True))
        console.print("[bold magenta]Average %diffs (y, y_hat):")
        console.print(Pretty(np.array(total_pred_sr).mean(), indent_guides=True))
        console.print(Rule(style="bold blue"))

    elif formulation=="y":

        errs = {
            "psnr": np.array(psnr_vals).mean(),
            "ssim": np.array(ssim_vals).mean(),
        }

        # avg_pred_errs = {key: value / num_samples for key, value in total_pred_errs.items()}
        # avg_baseline_errs = {key: value / num_samples for key, value in total_baseline_errs.items()}
        
        # write results to log
        # total_pred_errs.update(total_baseline_errs)
        # logger.log(**total_pred_errs)
        
        console = Console()
        console.print(Rule(f"[bold blue]Results for SR: {upsampling_ratio} | dataset: {dataset_name}"))
        # console.print("[bold magenta]Average %diffs (y, y_sparse):")
        # console.print(Pretty(avg_baseline_errs, indent_guides=True))
        console.print("[bold magenta]Average %diffs (y, y_hat):")
        console.print(Pretty(errs, indent_guides=True))
        console.print(Rule(style="bold blue"))

# --------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--ckpt_fp", type=str, default="")
    parser.add_argument("-fm", "--formulation", type=str, default="y")
    parser.add_argument("-ds", "--dataset", type=str, default="mos2-sef")
    parser.add_argument("-sr", "--upsampling_ratio", type=int, default=2)
    parser.add_argument("-er", "--exp_root", type=str, default="")
    parser.add_argument("-en", "--exp_name", type=str, default="")
    args = parser.parse_args()

    assert os.path.isfile(args.ckpt_fp)
    assert args.formulation in ["X", "y"]
    assert args.dataset in ["mos2-sef", "silicon", "sapphire"]
    args.upsampling_ratio = int(args.upsampling_ratio)
    assert args.upsampling_ratio in [2, 4, 8]
    
    eval(args.ckpt_fp, args.formulation, args.dataset, args.upsampling_ratio)