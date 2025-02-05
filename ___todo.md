1. Delete old model weights + create a better system for managing model ckpts over time.
    - e.g., how can we prevent a mountain of old model weights from building up?
2. Use test-cases to 100% guarentee that our dataloading does not...
    - A. have any data contamination
    - B. never flips a mask during data augmentation
3. Experiments
    - 3a. Surrogate Loss: {L1 (baseline), OLDER, OLDER + L1, sigmoid(OLDER) + L1}
    - 3b. Formulations: {p(y|y_sparse), p(y|X), p(y|y_sparse, X)}
4. Eval 3a, 3b