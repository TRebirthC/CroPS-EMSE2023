"""
This file is used to delete the invalid results.
Please use it with caution.
"""

import os
import shutil
import os.path

rootdir = "../results/rslt.save"
prefix = "pbsa"


def delete_invalid_folders(postfix):
    """
    This method is used to delete invalid folders whose name is end as postfix.

    Args:
        postfix (string): The postfix of the name of the invalid folders.
    """
    invalid_folder_name = prefix + postfix
    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:
            if dirname == invalid_folder_name:
                dir = parent + os.sep + dirname
                # dir = parent + os.sep + dirname + os.sep + invalid_folder_name + "-cpps-para-1000step-5run.pkl"
                exist = os.path.exists(dir)
                if exist:
                    shutil.rmtree(dir)
                    # os.remove(dir)


def delete_invalid_folders_in(inner):
    """
    This method is used to delete invalid folders whose name contains inner.

    Args:
        inner (string): The context in the name of the invalid folders.
    """
    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:
            if inner in dirname:
                shutil.rmtree(parent + os.sep + dirname)


if __name__ == "__main__":
    delete_invalid_folders_in("90d")
