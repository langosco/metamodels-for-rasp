import logging
import sys

def setup_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_data_logger(level=logging.DEBUG, logfile="train.log"):
    logger = logging.getLogger("data_logger")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger