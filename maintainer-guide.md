# Packaging and uploading to PyPi

More details can be found [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## Update the version in pyproject.toml.

Increment the minor or major version appropriately from the line indicated below in pyproject.toml.
````
version = "1.0.3"
````

## Build
````
python3 -m build
````

## Upload to PyPi

````
python3 -m twine upload --repository pypi dist/*
````
