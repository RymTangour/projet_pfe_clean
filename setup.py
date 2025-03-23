from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    
    HYPHEN_E_DOT = '-e .'
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ransomeware_detection_using_machine_learning',
    version='0.0.1',
    author='YoussefBaryoul, RymTangour', 
    author_email='baryouly@gmail.com, tangourrim@gmail.com',
    description='A machine learning project for ransomware detection.',
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    install_requires=get_requirements('requirements.txt'),  
)