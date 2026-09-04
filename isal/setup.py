from setuptools import find_packages, setup


setup(
    name="isal-humanoid",
    version="0.1.0",
    description="Interaction-Supervised Affordance Learning for humanoid locomotion",
    packages=find_packages(),
    install_requires=["robolab", "rsl-rl-lib"],
)
