import os
from pathlib import Path
import zipfile
from box import ConfigBox
import yaml
from box.exceptions import BoxValueError  
from ensure import ensure_annotations



@ensure_annotations
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    try:
        with open(path_to_yaml)as yaml_file:
            content=yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations        
def create_directories(path_to_directories: list , verbose=True):     
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"created directory at: {path}")

