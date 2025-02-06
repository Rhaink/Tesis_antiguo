from setuptools import setup, find_packages

setup(
    name="pulmo_align",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "Pillow",
    ],
    author="PulmoAlign Team",
    description="Biblioteca para el procesamiento y análisis de imágenes pulmonares",
    python_requires=">=3.6",
)
