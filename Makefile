ENV_NAME := openff-env

CONDA_ENV_RUN   = conda run --no-capture-output --name $(ENV_NAME)

.PHONY: env env-dev clean-nb format-nb run-nb

env:
	mamba create     --name $(ENV_NAME)
	mamba env update --name $(ENV_NAME) --file environment.yaml

env-dev:
	mamba create     --name $(ENV_NAME)
	mamba env update --name $(ENV_NAME) --file environment.yaml
	# Now, add dev dependencies
	mamba env update --name $(ENV_NAME) --file devtools/conda-envs/dev-env.yaml
	$(CONDA_ENV_RUN) pre-commit install || true

clean-nb:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

format-nb:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec nbqa ruff --fix {} --ignore E402 \;

run-nb:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;

run-nb-and-convert-to-md:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec jupyter nbconvert --to markdown --execute --output-dir notebooks-rendered {} \;
