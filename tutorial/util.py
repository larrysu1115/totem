# -*- coding: utf-8 -*-
"""Utility functions"""
import tensorflow as tf

def recreate_folder(folder_path):
    """Recreate folder"""
    if tf.gfile.Exists(folder_path):
        tf.gfile.DeleteRecursively(folder_path)
        tf.logging.info("Deleted existing model dir: %s", folder_path)
    tf.gfile.MakeDirs(folder_path)
