import logging

logging.basicConfig(
    level=logging.DEBUG,
    filemode="w",
    format=f"%(levelname)s-%(asctime)s-%(filename)s-%(funcName)s: %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
)
logger = logging.getLogger(__name__)
