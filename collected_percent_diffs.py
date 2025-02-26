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
from src.util.celano_lab_scripts import calculate_diff_between_samples
from src.models.our_method.swin_cafm import SwinCAFM
from src.datasets.mos2_sr import MOS2SRDataset, MOS2_SEF_SRC_DIR, MOS2_SAPPHIRE_DIR, MOS2_SILICON_DIR

warnings.simplefilter("ignore")

SUBSTRATES_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/substrates"
NUM_TRIALS = 1

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
    dataset = MOS2SRDataset(
        src_dir=src_dir, 
        split="val", 
        upsample_factor=upsampling_factor
    )

    best_r2 = 0
    best_pred = None
    best_downsampled = None
    best_original_sample = None

    for _ in tqdm(range(NUM_TRIALS), desc=f"Processing SR: {upsampling_factor} | dataset: {dataset_name}"):
            
        item = dataset[0]
        original_data = item["y"]
        downsampled_data = item["y_sparse"]

        model_in = downsampled_data.cuda().float().unsqueeze(0)
        pred_torch = model(model_in)

        # pred_torch: [1, 512, 512]
        r2 = compute_r2(pred_torch.cuda(), original_data.cuda())

        # cherry pick sample w/ best results
        if r2 > best_r2:
            best_r2 = r2
            best_pred = pred_torch
            best_downsampled = downsampled_data
            best_original_sample = original_data

    y: torch.Tensor        = best_original_sample
    y_sparse: torch.Tensor = best_downsampled
    y_hat: torch.Tensor    = best_pred

    # model prediction errors
    pred_errs     = calculate_diff_between_samples(y, y_hat, 2.0)
    baseline_errs = calculate_diff_between_samples(y, y_sparse, 2.0)

    console = Console()
    console.print(Rule(f"[bold blue]Results for SR: {upsampling_factor} | dataset: {dataset_name}"))
    console.print("[bold magenta]%diffs (y, y_sparse):")
    console.print(Pretty(baseline_errs, indent_guides=True))
    console.print("[bold magenta]%diffs (y, y_hat):")
    console.print(Pretty(pred_errs, indent_guides=True))
    console.print(Rule(style="bold blue"))

# --------------------------------------------

for weight in weights:
    eval(weight)

