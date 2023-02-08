from abc import ABC, abstractmethod
import logging


class AbstractLogger(ABC):

    def __init__(self, path: str, level: str = "INFO"):
        class_name = type(self).__name__
        logging.basicConfig(filename=path, level=level)
        self.logger = logging.getLogger(class_name)

    @abstractmethod
    def log_set_up(self, investigated_object: object):
        self.logger.info(f"SET UP OF: {investigated_object}")


class PtLogger(AbstractLogger):

    def log_set_up(self, investigated_object, ):
        super(PtLogger, self).log_set_up(investigated_object)
        # self.logger.info(f"kwargs: {self.kwargs}")
        # self.logger.info(f"parameter: {self.varied_parameter}")
        # self.logger.info(f"range of values: {self.parameter_range}")