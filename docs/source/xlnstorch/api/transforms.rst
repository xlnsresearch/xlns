.. currentmodule:: xlnstorch.transforms

.. _transforms-doc:

Transforms
==========

The ``xlnstorch.transforms`` submodule provides a data transformation class,
``ToLNSTensor``, which converts various input formats (like images and tensors)
into LNSTensor objects. It is analogous to torchvision's
:class:`torchvision.transforms.ToTensor` transform.

This submodule requires the ``torchvision`` package to be installed, which can
be done via pip:

.. code-block:: bash

    pip install torchvision

.. autosummary::
    :toctree: generated/transforms
    :nosignatures:

    ToLNSTensor

.. note::

    By default, the ToLNSTensor transform will only wrap floating point tensors.
    If you want to wrap all tensor types, set the `wrap_all` parameter to `True`.

Quick Start
-----------

Here's a simple example of using the LNS transforms with image data:

.. code-block:: python

    import xlnstorch as xltorch
    from xlnstorch.transforms import ToLNSTensor
    from PIL import Image

    # Create transform to convert image to LNSTensor
    transform = ToLNSTensor(f=8, device="cpu")

    # Load and transform an image
    image = Image.open("example.jpg")
    lns_tensor = transform(image)
    print(type(lns_tensor))  # <class 'xlnstorch.LNSTensor'>

For use with PyTorch datasets:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import xlnstorch as xltorch
    from xlnstorch.transforms import ToLNSTensor

    transform = ToLNSTensor(f=8)

    # Download a dataset and apply the transform
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Get the first batch of data
    data, target = next(iter(dataloader))
    print(data.shape) # torch.Size([16, 1, 28, 28])
    print(type(data))  # <class 'xlnstorch.LNSTensor'>
    print(target.shape) # torch.Size([16])
    print(type(target))  # <class 'torch.Tensor'>