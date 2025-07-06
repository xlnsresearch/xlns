import torch
from .. import LNSTensor

class LNSFunction(torch.autograd.Function):
    """
    Base class for LNS operations that require custom forward and backward methods.
    This class should be subclassed for specific LNS operations.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Forward pass for the LNS operation.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass for the LNS operation.
        Should be implemented in subclasses.
        """
        raise NotImplementedError("Backward method must be implemented in subclasses.")
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Applies the LNS operation defined by this class.
        This method is used to call the forward and backward methods.

        Note that any keyword arguments passed to this method will raise an error,
        as `torch.autograd.Function` does not support keyword arguments. Instead,
        use positional arguments only.

        In addition, this method converts any `LNSTensor` arguments to their
        internal representation (i.e., the underlying tensor) before calling the
        forward method. This is necessary for LNSTensor internal behavior.
        """
        # This check is also performed in the base class, but we do it here too
        # in case PyTorch decides to change the behavior of the apply method.
        if kwargs:
            raise ValueError("torch.autograd.Function does not support keyword arguments. Please use positional arguments only.")

        # Convert LNSTensor arguments to internal representation. This is necessary
        # because the autograd.Function expects tensors, not LNSTensor objects.
        internal_args = []
        for arg in args:
            if isinstance(arg, LNSTensor):
                internal_args.append(arg._lns)
            else:
                internal_args.append(arg)

        # call the forward method of the class with the internal arguments
        result = super().apply(*internal_args)

        # get all output tensors and store them in a tuple
        if isinstance(result, torch.Tensor):
            result_tuple = (result,)
        elif isinstance(result, (tuple, list)):
            result_tuple = tuple(result)
        else:
            result_tuple = tuple()

        # we register hooks on each input to each output for gradient accumulation
        for output in result_tuple:

            # only register hooks for tensors outputs that require gradients
            if not (isinstance(output, torch.Tensor) and output.requires_grad):
                continue

            # get the gradient edge for the output tensor.
            edge = torch.autograd.graph.get_gradient_edge(output)

            for i in range(len(args)):

                # only register hooks for LNSTensor inputs with gradients
                if not (isinstance(args[i], LNSTensor) and args[i].requires_grad):
                    continue

                # track operation by registering a hook on the input tensor
                # to use custom addition logic each time it receives a gradient
                args[i]._track_operation(edge, i)

        return result