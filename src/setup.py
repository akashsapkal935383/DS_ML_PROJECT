from setuptools import find_packages,setup
from typing import List


def get_requirements(file_path:str)->List[str]:

    requiremenets=[]

    with open (file_path) as file_obj:
        requiremenets=file_obj.readline()
        requiremenets=[ req.replace ("\n","")for req in requiremenets]

    return requiremenets






setup(
     
     name='student_ml_project',
     version='0.0.1',
     author="Newgen infotech",
     author_email='Balasaheb.aiml@gmail.com',
     packages=find_packages(),
     install_requires=get_requirements('requirements.txt')
)