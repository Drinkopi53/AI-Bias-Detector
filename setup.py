from setuptools import setup, find_packages

setup(
    name="ai_bias_detector",
    version="0.1.0",
    packages=find_packages(),
    author="Jules",
    description="A Python library to detect bias in AI models and datasets.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/user/ai-bias-detector", # Placeholder URL
    install_requires=[
        "pandas",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Placeholder License
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
