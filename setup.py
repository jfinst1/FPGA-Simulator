from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fpga-simulator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced FPGA Simulator with GPU, Quantum, and ML Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fpga-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: System :: Hardware",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "flask>=2.0.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "quantum": ["qiskit>=0.39.0", "qiskit-aer>=0.11.0"],
        "ml": ["scikit-learn>=1.0.0", "scipy>=1.7.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "all": [
            "cupy-cuda11x>=10.0.0",
            "qiskit>=0.39.0",
            "qiskit-aer>=0.11.0",
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fpga-sim=fpga_simulator:main",
            "fpga-web=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.css", "*.js"],
    },
)