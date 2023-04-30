### TLDR

#### PRECOMMIT

- `pip install pre-commit pip-tools black flake8`
- create/update .pre-commit-config.yaml
- `pre-commit install`

### NEXT STEPS

- enable custom wrappers for Encoding, Scaling, and Model
- enable optuna in the process [ask&tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) [hyperband](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner)
- create regression report to gather all diagnostics for both **train and test data**
- develop a cml pipeline
- adjust plot names in save_fig string
- enable save of the cv_results in a meaningful manner
