# from Experiments.helpers.import_env import *
import os
import sys
from masthay_helpers.import_env import *

include_paths = ["masthay_helpers"]

# run_make_files(omissions)

init_modules(os.getcwd(), root=True, inclusions=include_paths, unload=True)