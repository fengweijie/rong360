#coding:utf-8
import os
import configparser
import re

def get_section_value(hos_conf,section):
    '''
    获取每个section的元素
    :param hos_conf:
    :param section:
    :return:
    '''
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    params = {}
    for key, value in hos_conf.items(section):
        if pattern.match(value):
            if type(eval(value))==int:
                params[key] = (int)(value)
            if type(eval(value))==float:
                params[key] = (float)(value)
        else:
            params[key] = value
    return params


def read_config_ini(file_path):
    '''
    读取配置文件
    :param file_path:
    :return:
    '''
    hos_conf = configparser.ConfigParser()
    hos_conf.read(file_path)
    section_value = {}
    for section in hos_conf.sections():
        section_value[section] = get_section_value(hos_conf, section)
    return section_value