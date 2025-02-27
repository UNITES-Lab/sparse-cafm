import rich
import torch
import warnings
import numpy as np

from rich.console import Console
from rich.rule import Rule
from rich.pretty import Pretty
from pprint import pprint
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.util.celano_lab_scripts import calculate_diff_between_samples
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import MOS2SRDataset, MOS2_SEF_SRC_DIR, MOS2_SAPPHIRE_DIR, MOS2_SILICON_DIR

warnings.simplefilter("ignore")

SUBSTRATES_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates"
NUM_TRIALS = 256

weights = glob(SUBSTRATES_DIR + "/*/*/*/*_best.pth")
console = Console()


@torch.no_grad()
def compute_r2(X_new: torch.Tensor, X_ref: torch.Tensor) -> float:
    X_new = X_new.float()
    X_ref = X_ref.float()
    X_ref_mean = torch.mean(X_ref)
    rss = torch.sum((X_new - X_ref) ** 2)
    tss = torch.sum((X_ref - X_ref_mean) ** 2)
    r2 = 1 - (rss / tss)
    return r2.item()


@torch.no_grad()
def eval(fp: str):

    # load model obj
    model: SwinCAFM = torch.load(fp).cuda().float(); model.eval();
    
    dataset_name = fp.split("/")[-4]
    src_dir = ""
    if dataset_name == "mos2-sef": src_dir = MOS2_SEF_SRC_DIR
    if dataset_name == "sapphire": src_dir = MOS2_SAPPHIRE_DIR
    if dataset_name == "silicon": src_dir = MOS2_SILICON_DIR
    upsampling_factor = int(fp.split("SR-")[1:][0][0])

    # grab the right dataset obj
    dataset = MOS2SRDataset(src_dir=src_dir, split="val", upsample_factor=upsampling_factor, steps_per_epoch=NUM_TRIALS)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    # best_r2 = 0
    # best_pred = None
    # best_downsampled = None
    # best_original_sample = None

    total_pred_errs = None
    total_baseline_errs = None
    num_samples = 0

    for batch in tqdm(data_loader, desc=f"Processing SR: {upsampling_factor} | dataset: {dataset_name}"):
        y = batch["y"]
        y_sparse = batch["y_sparse"]

        y = y.cuda().float()
        y_sparse = y_sparse.cuda().float()
        y_hat = model(y_sparse)

        # TOOD: scale + shift
        y = (y - dataset.current_maps_mean) / (dataset.current_maps_std)
        y_hat = (y_hat - dataset.current_maps_mean) / (dataset.current_maps_std)
        y_sparse = (y_sparse - dataset.current_maps_mean) / (dataset.current_maps_std)

        for i in range(y_hat.size(0)):
            
            sample_y = y[i]
            sample_y_sparse = y_sparse[i]
            sample_y_hat = y_hat[i]

            current_pred_errs = calculate_diff_between_samples(sample_y, sample_y_hat, 2.0)
            current_baseline_errs = calculate_diff_between_samples(sample_y, sample_y_sparse, 2.0)

            if total_pred_errs is None:
                total_pred_errs = {key: value for key, value in current_pred_errs.items()}
                total_baseline_errs = {key: value for key, value in current_baseline_errs.items()}
            else:
                for key in current_pred_errs:
                    total_pred_errs[key] += current_pred_errs[key]
                for key in current_baseline_errs:
                    total_baseline_errs[key] += current_baseline_errs[key]
                    
            num_samples += 1

    avg_pred_errs = {key: value / num_samples for key, value in total_pred_errs.items()}
    avg_baseline_errs = {key: value / num_samples for key, value in total_baseline_errs.items()}

    breakpoint()

    console = Console()
    console.print(Rule(f"[bold blue]Results for SR: {upsampling_factor} | dataset: {dataset_name}"))
    console.print("[bold magenta]Average %diffs (y, y_sparse):")
    console.print(Pretty(avg_baseline_errs, indent_guides=True))
    console.print("[bold magenta]Average %diffs (y, y_hat):")
    console.print(Pretty(avg_pred_errs, indent_guides=True))
    console.print(Rule(style="bold blue"))

# --------------------------------------------

for weight in weights:
    eval(weight)

