# setup.py
from setuptools import setup, find_packages

setup(
    name="radiotherapy",
    version="0.1",
    # list each module in sourcecode/ as a standalone py_module:
    py_modules=[
        "gui",
        "main",
        "system_manager",
        "data_management",
        "training_pipeline",
        "inference_module",
        "visualization",
        "storage",
        "parameter_manager"
    ],
    package_dir={"": "sourcecode"},
    install_requires=[
        "PySide6", "monai", "torch", "nibabel", "numpy"
    ],
    entry_points={
        "gui_scripts": [
            "radiotherapy-gui = gui:main",
        ]
    }
)