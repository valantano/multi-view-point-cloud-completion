from setuptools import setup, find_packages

setup(
    name="PoinTrTest",  # Name of the overall package
    version="0.1.0",  # Package version
    packages=find_packages(),  # Automatically find sub-packages (like project1, project2)
    install_requires=[],  # Add dependencies here if needed
    description="A collection of submodule Python projects for",
    author="Valentino Geuenich",
    author_email="valentino.geuenich@rwth-aachen.de",
    url="https://github.com/your-repo-url",  # Replace with your repo URL if applicable
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify Python version compatibility
)