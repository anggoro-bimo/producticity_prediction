from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """The function will return a list of python libraries and module requirements
    for this project.
    """
    requirements = []
    with open(file_path) as file_obj:
        requiremnents = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements] # Replace the new line space with blank

        # Connecting requirements to setup.py
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
setup(
name = 'Productivity Prediction of Garment Factory Employees',
version = '1.0.0',
author = 'Bimo Anggoro',
author_email = 'bimoanggorop@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)