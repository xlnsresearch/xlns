# Packaging and uploading to PyPi

More details can be found [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

Currently, we don't use a tool like [cibuildwheel](https://github.com/pypa/cibuildwheel) to compile
wheels for each python version and platform. This means we must upload only the source code to pypi
and allow the user to compile the C++ extensions on their machine.

## Update the version in pyproject.toml.

Increment the minor or major version appropriately from the line indicated below in pyproject.toml.
````
version = "1.0.0"
````

## Build
````
python3 -m build ./xlnstorch
````

## Upload only the source code to PyPi

````
python3 -m twine upload --repository pypi xlnstorch/dist/*.tar.gz
````