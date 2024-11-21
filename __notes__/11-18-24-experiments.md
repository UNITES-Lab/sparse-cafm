# basic experimental procedure

0. metadata
1. complete config / condition
2. results

# format

- {YYYY-MM-DDTHH:mm:ss}-{model_name}-{dataset_name}-{exp_cond}
    - config.yaml
        - should include a model, hyper-params, and data section
    - log.csv
    - notes.md

# experiment obj

- instance of logger
- some way to record all hps