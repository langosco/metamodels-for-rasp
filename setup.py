from setuptools import setup

setup(name='metamodels-for-rasp',  # For pip. E.g. `pip show`, `pip uninstall`
      version='0.0.1',
      author="Lauro Langosco",
      description="Training code for *Towards Meta-Models for Interpretability*",
      packages=["metamodels_for_rasp"], # For python. E.g. `import python_template`
      install_requires=[
        "numpy",
        "jax",
        "chex",
        "jax",
        "numpy",
        "optax",
        "wandb",
        "einops",
        "orbax-checkpoint",
        "flax"
      ],
)
