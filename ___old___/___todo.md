# **TODO**
---
1. Train a multi-headed OLDER-surrogate; verify that this model works very well before moving on
2. Using OLDER-surrogate to perform these ablations: {baseline, OLDER, OLDER + L1, ...}

# **Notes**
---
1. We train an OLDER surrogate that approximates the difference in characterization between two current-maps
    - We want this surrogate to encode meaningful features of each current map
    - We should scale up the size of OLDER pre-training...
    - What would happen if we mixed in ImageNet data or something?
2. 