.. currentmodule:: xlnstorch

.. _autograd-graphs-doc:

Autograd Graph Visualization
============================

This section covers tools for visualizing PyTorch's autograd computation graphs
when using LNS tensors. These visualizations are particularly useful for debugging
gradient flows, understanding computational graph structure, and analyzing parameter
relationships in LNS-based models.

Basic Graph Visualization
-------------------------

:func:`viz.graph.make_autograd_graph` creates graphviz visualizations of PyTorch's
autograd computation graphs, with special support for LNS tensors.

**Simple Computation Graph**

.. code-block:: python

    import xlnstorch as xltorch
    from xlnstorch.viz.graph import make_autograd_graph

    # Simple computation graph
    x = xltorch.randn(2, 2, requires_grad=True)
    y = x.pow(2).sum()

    graph = make_autograd_graph(y)
    graph.view()  # Opens in default viewer

**Multi-Variable Operations**

.. code-block:: python

    # More complex operations
    x = xltorch.randn(3, 3, requires_grad=True)
    y = xltorch.randn(3, 3, requires_grad=True)

    # Perform operations
    z = x * y + x.pow(2)
    loss = z.sum()

    # Generate autograd graph
    graph = make_autograd_graph(loss, params={'x': x, 'y': y})
    graph.render('autograd_graph', format='svg')

Advanced Features
-----------------

**Model Parameter Highlighting**

.. code-block:: python

    # More complex graph with parameter highlighting
    model = xltorch.nn.LNSLinear(10, 5)
    x = xltorch.randn(3, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()

    # Highlight model parameters in the graph
    graph = make_autograd_graph(
        loss,
        params={
            'input': x,
            'weight': model.weight,
            'bias': model.bias
        },
        show_saved=True,
        leaf_color='lightblue',
        node_color='lightgrey'
    )
    
    graph.render('model_graph', format='svg')

**Custom Node Styling**

.. code-block:: python

    # Create a computation with custom styling
    x1 = xltorch.randn(5, requires_grad=True)
    x2 = xltorch.randn(5, requires_grad=True)

    # Chain of operations
    intermediate = x1 * x2
    result = torch.exp(intermediate).sum()

    graph = make_autograd_graph(
        result,
        params={'x1': x1, 'x2': x2, 'intermediate': intermediate},
        leaf_color='lightgreen',
        node_color='lightcoral',
        output_color='gold',
        show_saved=True
    )

    graph.render('styled_graph', format='pdf')

Debugging Gradient Flows
-------------------------

**Identifying Gradient Bottlenecks**

.. code-block:: python

    # Create a model with potential gradient issues
    class TestModel(xltorch.nn.LNSModule):
        def __init__(self):
            super().__init__()
            self.layer1 = xltorch.nn.LNSLinear(10, 20)
            self.layer2 = xltorch.nn.LNSLinear(20, 10)
            self.layer3 = xltorch.nn.LNSLinear(10, 1)

        def forward(self, x):
            x = self.layer1(x)
            x = torch.nn.functional.relu(x)
            x = self.layer2(x)
            x = torch.nn.functional.sigmoid(x)
            x = self.layer3(x)
            return x

    model = TestModel()
    x = xltorch.randn(5, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()

    # Create comprehensive parameter dictionary
    params = {'input': x}
    for name, param in model.named_parameters():
        params[name] = param

    graph = make_autograd_graph(
        loss, 
        params=params,
        show_saved=True
    )
    graph.render('model_debug', format='svg')

**Analyzing Gradient Flow in Recurrent Operations**

.. code-block:: python

    # Example with recurrent-like structure
    x = xltorch.randn(1, requires_grad=True)
    states = [x]
    
    # Simulate recurrent operations
    for i in range(3):
        next_state = states[-1] * 0.5 + xltorch.randn(1, requires_grad=True)
        states.append(next_state)
    
    final_output = sum(states).sum()
    
    # Create parameter dict for all states
    params = {f'state_{i}': state for i, state in enumerate(states)}
    
    graph = make_autograd_graph(final_output, params=params)
    graph.render('recurrent_flow', format='svg')

Practical Examples
------------------

**Comparing LNS vs Float32 Graphs**

.. code-block:: python

    # Create identical operations in LNS and float32
    def create_computation(tensor_type):
        if tensor_type == 'lns':
            x = xltorch.randn(3, 3, requires_grad=True)
            y = xltorch.randn(3, 3, requires_grad=True)
        else:
            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
        
        z = x @ y  # Matrix multiplication
        w = z.pow(2)
        loss = w.sum()
        
        return loss, {'x': x, 'y': y}
    
    # LNS computation
    lns_loss, lns_params = create_computation('lns')
    lns_graph = make_autograd_graph(lns_loss, params=lns_params)
    lns_graph.render('lns_computation', format='png')
    
    # Float32 computation  
    float_loss, float_params = create_computation('float32')
    float_graph = make_autograd_graph(float_loss, params=float_params)
    float_graph.render('float32_computation', format='png')

**Training Loop Visualization**

.. code-block:: python

    # Visualize one step of training
    model = xltorch.nn.LNSLinear(5, 1)
    optimizer = xltorch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    x = xltorch.randn(10, 5, requires_grad=True)
    target = xltorch.randn(10, 1)
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()
    
    # Create graph before backward pass
    params = {
        'input': x,
        'target': target,
        'prediction': prediction,
        'weight': model.weight,
        'bias': model.bias
    }
    
    graph = make_autograd_graph(loss, params=params)
    graph.render('training_step', format='svg')

Export Formats and Integration
------------------------------

Graphs can be exported in various formats for different use cases:

.. code-block:: python

    # Different export formats
    x = xltorch.randn(2, 2, requires_grad=True)
    y = (x * 2).sum()
    
    graph = make_autograd_graph(y, params={'x': x})
    
    # Vector formats (scalable)
    graph.render('graph_vector', format='svg')
    graph.render('graph_vector', format='pdf')
    
    # Raster formats (fixed resolution)
    graph.render('graph_raster', format='png')
    graph.render('graph_raster', format='jpg')
    
    # Source code formats
    graph.render('graph_source', format='dot')  # Graphviz source
    
    # Direct viewing
    graph.view()  # Opens with system default viewer