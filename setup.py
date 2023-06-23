from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="FLUID_project_upprpo",
    version='1.1.1',
    author="VeLoR",
    author_email="",
    description="",
    url = "https://github.com/AxeVal/Fluid",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'opencv-python', 'pygame'],
    keywords=['pypi', 'cicd', 'python'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows"
    ]
)