from importlib import util
import os

# Get the path to the root optillm.py
spec = util.spec_from_file_location(
    "optillm.root",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "optillm.py")
)
module = util.module_from_spec(spec)
spec.loader.exec_module(module)

# Export the main function
main = module.main