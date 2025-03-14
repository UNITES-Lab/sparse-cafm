# TODO: implement

import os
import yaml
from util.logger import Logger

class Experiment:
    """
    The `Experiment` class defines a universal experiment format.
    
    Encapsulates complete experimental setting and result data.
    """
    
    def __init__(self, config_fp: str, logger: Logger, experiment_dir: str) -> None:
        """
        """
        
        assert config_fp.endswith('.yaml')
        assert os.path.isfile(config_fp)
        assert os.path.isdir(experiment_dir)
        try:
            self.config = yaml.load(open(config_fp, 'r'), Loader=yaml.Loader)
        except:
            raise Exception(f'Failed to load config file @: {config_fp}')
        self.logger = logger
        
    def update(self):
        pass