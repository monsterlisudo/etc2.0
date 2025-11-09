"""
Traffic Fingerprinting Network 安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements.txt
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="traffic-fingerprinting",
    version="1.0.0",
    author="MiniMax Agent",
    author_email="support@minimax.ai",
    description="基于深度学习的网络流量分类和指纹识别系统",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/traffic-fingerprinting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "pre-commit>=2.13",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=5.0",
        ],
        "perf": [
            "thop>=0.1.0",
            "tqdm>=4.62.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-fingerprint=traffic_classifier.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/your-username/traffic-fingerprinting/issues",
        "Source": "https://github.com/your-username/traffic-fingerprinting",
        "Documentation": "https://traffic-fingerprinting.readthedocs.io/",
    },
    keywords=[
        "traffic-analysis",
        "network-security", 
        "deep-learning",
        "pytorch",
        "machine-learning",
        "reinforcement-learning",
        "network-classification",
        "website-fingerprinting",
    ],
)