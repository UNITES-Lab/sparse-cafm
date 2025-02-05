1. Delete old model weights + create a better system for managing model ckpts over time.
    - e.g., how can we prevent a mountain of old model weights from building up?
2. Use test-cases to 100% guarentee that our dataloading does not...
    - A. have any data contamination
    - B. never flips a mask during data augmentation
3. Re-run evals for all checkpoints evaled with wrong sparse prior