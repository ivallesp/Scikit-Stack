# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'
import unittest

from Stacker.file_tools import *


class TestFileTools(unittest.TestCase):
    def test_generate_path_for_copy(self):
        path = make_sure_do_not_replace("../demo/file_tools/foo.txt")
        self.assertNotEqual(path, "../demo/file_tools/foo.txt")
        path = make_sure_do_not_replace("../demo/file_tools/foo_2.txt")
        self.assertEqual(path, "../demo/file_tools/foo_2.txt")
        path = make_sure_do_not_replace("../demo/file_tools/foo_3.txt")
        self.assertEqual(path, "../demo/file_tools/foo_3_copy3.txt")
