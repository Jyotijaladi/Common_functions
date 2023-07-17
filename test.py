import inspect
import Python_lib

def has_arguments(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    return len(parameters) > 0

# Get the functions from the module
module_functions = inspect.getmembers(Python_lib, inspect.isfunction)

# Check if each function has arguments
for function_name, function_obj in module_functions:
    if has_arguments(function_obj):
        print(f"The function '{function_name}' has arguments.")
