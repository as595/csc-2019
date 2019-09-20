---
layout: page
permalink: /hint2/
---

When you test against your validation dataset you don't want your network to use the dat from training, so you must set the mode of your network to <code>evaluation</code>

```python
net = net.eval()
```

when you're done you can set it back to training mode:

```python
net = net.train()
```

You also don't want the gradient to be calculated when you calculate the loss for the validation dataset, so you must make your calculation conditional:

```python
with torch.no_grad():

    blah blah
    my code ...
```


