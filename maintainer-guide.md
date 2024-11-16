# Packaging and uploading to PyPi

More details can be found [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## Build
````
python3 -m build
````

## Upload to PyPi

````
python3 -m twine upload --repository pypi dist/*
````
