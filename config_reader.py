import configparser
import os
import utils
import re

class My_Config_Parser(configparser.ConfigParser):
    #def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)

    def get_as_slice(self, num_cols, *args, **kwargs):
        raw_get = self.get(*args, **kwargs)
        matcher = re.search('(-?\d*):(-?\d*)', raw_get)

        if ':' in raw_get:
            if matcher.group(1) and matcher.group(2):
                return slice(int(matcher.group(1)), int(matcher.group(2)))
            elif matcher.group(2):
                return slice(0, int(matcher.group(2)))
            else:
                return slice(int(matcher.group(1), num_cols))

        else:
            return int(raw_get)

    def get_rel_path(self, *args, **kwargs):
        raw_get = self.get(*args, **kwargs)
        if not raw_get:
            return ""
        if raw_get.startswith('/'):
            return raw_get

        return utils.abs_path_of(raw_get)



def read_config(path):
    config = My_Config_Parser(inline_comment_prefixes=['#'], interpolation=configparser.ExtendedInterpolation())
    config.read(path)

    return config


def get_task_sections(config):
    return {section_name: config[section_name] for section_name in config.sections() if
                section_name.startswith("TASK")}


# config = read_config("config/default.ini")
# print(config.get_slice("FEATURES","columns"))
# print ([1,2,3][config.get_slice("FEATURES","columns")])
