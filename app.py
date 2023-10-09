from detection.logger import logging
from detection.exception import AppException
import sys

logging.info("Welcome to my custom logger")

try:
    a = 33/"a"
except Exception as error:
    raise AppException(error, sys)