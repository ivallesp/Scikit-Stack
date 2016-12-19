# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'
import os


def make_sure_do_not_replace(path):
    """
    This function has been created for those cases when a file has to be generated and we want to be sure that we don't
    replace any previous file. It generates a different filename if a file already exists with the same name
    :param path: path to be checked and adapted (str or unicode)
    :return: path (str or unicode)
    """
    base, fileName = os.path.split(path)
    file_name, file_ext = os.path.splitext(fileName)
    i = 1
    while os.path.exists(path):
        path = os.path.join(base, file_name + "_copy%s" % i + file_ext)
        i += 1
    return path
