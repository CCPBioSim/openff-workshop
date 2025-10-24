ENV_NAME := openff-env

CONDA_ENV_RUN   = conda run --no-capture-output --name $(ENV_NAME)

.PHONY: env env-dev clean-nb format-nb run-nb create-student-nb

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

create-student-nb:
	$(CONDA_ENV_RUN) python devtools/scripts/remove_notebook_solutions.py notebooks_with_solutions/small_molecule_parameterisation.ipynb notebooks/small_molecule_parameterisation.ipynb
	$(CONDA_ENV_RUN) python devtools/scripts/remove_notebook_solutions.py notebooks_with_solutions/protein_ligand_complex_parameterisation_and_md.ipynb notebooks/protein_ligand_complex_parameterisation_and_md.ipynb

run-nb:
	$(CONDA_ENV_RUN) find notebooks_with_solutions -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;

run-nb-and-convert-to-md:
	$(CONDA_ENV_RUN) python devtools/scripts/execute_and_convert_notebooks.py \
		--input-dir notebooks_with_solutions --output-dir notebooks-rendered \
		--skip-tag ci_skip
