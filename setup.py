from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="FLUID",
    version='{{VERSION_PLACEHOLDER}}',
    author="VeLoR",
    author_email="",
    description="",
    url = "https://github.com/AxeVal/Fluid",
    long_description_content_type="",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'opencv-python', 'pygame'],
    keywords=['pypi', 'cicd', 'python'],
    classifiers=[
        "Development Status :: 1 - Working",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows"
    ]
)