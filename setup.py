from setuptools import setup, find_packages

setup(
    name="la_fat",
    version="0.2.0",
    description="LA Fat Segmentation — Deep-Module Epicardial Adipose Tissue Analysis Pipeline",
    author="LA Fat Segmentation Contributors",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "TotalSegmentator>=2.0.0",
        "SimpleITK",
        "scipy",
        "scikit-image",
        "matplotlib",
        "scikit-learn",
        "PyYAML",
        "nibabel",
        "pandas",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "la-fat=la_fat.cli:main_cli",
        ],
    },
)
