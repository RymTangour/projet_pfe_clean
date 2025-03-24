import os
import sys
import logging
import inspect

def get_logger():
   
    frame = inspect.stack()[1]
    filename = os.path.basename(frame.filename)
    file_name_without_ext = os.path.splitext(filename)[0]
    
   
    module = inspect.getmodule(frame[0])
    module_name = module.__name__
    
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
   
    log_filepath = os.path.join(log_dir, f"{file_name_without_ext}_logs.log")
    
   
    logging_str = "[%(asctime)s:%(levelname)s:%(module)s:%(name)s:%(message)s]"
    
    
    logger = logging.getLogger(module_name)
    
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(logging.Formatter(logging_str))
        
       
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(logging_str))
        
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    
        logger.propagate = False
    
    return logger