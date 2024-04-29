
import logging
import config

class Module_Class:
    def __init__(self,module):
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOGS_FILE),
            logging.StreamHandler(),
        ],
        )
        self.LOGGER = logging.getLogger(__name__)
        self.module = module
        self.paths = None
        self.check_config_class()

    def check_config_class(self):
        if self.__class__.__name__ not in self.module.keys():
            raise NotImplementedError(f"Cannot find {self.__class__.__name__}")

        self.paths = self.module[self.__class__.__name__]
