#coding:utf-8
import logging
import sys

def mylog():
    FORMATTER = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #创建日志处理器,并将日志级别设置为DEBUG
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    #创建流处理器handler,并设置其日志级别韦Debug
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    #创建流处理器
    formatter = logging.Formatter(FORMATTER)
    handler.setFormatter(formatter)

    #为日志器添加handler
    logger.addHandler(handler)
    return logger

logger = mylog()

