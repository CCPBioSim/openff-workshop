ENV_NAME := openff-env

CONDA_ENV_RUN   = conda run --no-capture-output --name $(ENV_NAME)

.PHONY: install clean format run

env:
	mamba create     --name $(ENV_NAME)
	mamba env update --name $(ENV_NAME) --file environment.yaml -y
	# Now, add dev dependencies
	mamba env update --name $(ENV_NAME) --file devtools/conda-envs/dev-env.yaml -y
	$(CONDA_ENV_RUN) pre-commit install || true

clean:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

format:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec nbqa ruff --fix {} --ignore E402 \;

run:
	$(CONDA_ENV_RUN) find . -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;
