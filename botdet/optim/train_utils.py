import os
import inspect
import logging
import time
import math


def logging_config(logger_name, folder=None, name=None, filemode='w',
                   level=logging.INFO, console_level=logging.DEBUG, no_console=False):
    """Config the logging.
    Args:
        logger_name (str) : name of the logger
        folder (str, optional): logging file folder
        name (str, optional): logging file name
        filemode (str, optional): logging file mode
        level (int, optional): file logging level
        console_level (int, optional): console logging level
        no_console (bool, optional): whether to disable the console log
    Returns:
        logger (logging.Logger): Logger object
    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    # Remove all the current handlers (loggers of the same name cannot be recreated)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.handlers = []
    logpath = os.path.join(folder, name + '.log')
    print('All Logs will be saved to {}'.format(logpath), flush=True)
    logger.setLevel(min(level, console_level))
    # create file handler
    f_handler = logging.FileHandler(logpath, mode=filemode)
    f_handler.setLevel(level)
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    if not no_console:
        # Initialize the console logging
        c_handler = logging.StreamHandler()
        c_handler.setLevel(console_level)
        # c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_format = logging.Formatter('%(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
    return logger


def time_since(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    if h == 0:
        if m == 0:
            return '%ds' % s
        else:
            return '%dm %ds' % (m, s)
    else:
        return '%dh %dm %ds' % (h, m, s)
