from inpainting import InpaintingEvaluationDataset

dataset = InpaintingEvaluationDataset(
    pad_img_to_mod_by=8,
)
dataset.__getitem__(0)
breakpoint()