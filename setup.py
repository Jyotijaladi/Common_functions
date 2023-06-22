from setuptools import setup,find_packages
from typing import List
REQUIREMENT_FILE_NAME="requirements.txt"
Project_Name="PYPI PUBLISHING"
Version="0.0.0.1"
AUTHOR="Abhishek"
DESCRIPTION="Experiment For publishing"
PACKAGES=["PYPI PUBLISHING"]


def get_requirement_list()->List[str]:
    
    """

    Description : This function is going to return list of requirement
    mentioned in requirements.txt file

    Return : This function is going to return a file which contain name
    of libraries mentioned in requirements.txt file     

    """
    with open(REQUIREMENT_FILE_NAME) as req_File:
        req_List=req_File.readlines() 
        req_File.close()
   
    req_List=req_List[:-1]
   
    return req_List






setup(

    name=Project_Name,
    version=Version,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    


)

if __name__=="__main__":
    print(get_requirement_list())