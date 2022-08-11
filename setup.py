from setuptools import find_packages, setup


setup(
    name="morl_baselines",
    description="Implementations of multi-objective reinforcement learning (MORL) algorithms.",
    version="0.1",
    packages=[package for package in find_packages() if package.startswith("morl_baselines")],
    install_requires=[
        "gym==0.24.1",  # Fixed version due to breaking changes in 0.25
        "numpy",
        "torch>=1.11",
        "pymoo"
    ],
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    author="Lucas Alegre",
    url="https://github.com/LucasAlegre/morl_baselines",
    author_email="lucasnale@gmail.com",
    keywords="reinforcement-learning-algorithms multi-objective reinforcement-learning machine-learning mo-gym gym morl baselines toolbox python data-science",
    license="MIT",
    long_description_content_type="text/markdown",
)
