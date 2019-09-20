
---
layout: page
permalink: /hint1/
---

To split out a validation set the first thing you need to decide is what fraction of your training data you want to use, for example 20%:

```python
valid_size = 0.2
```

you then need to randomly select 20% of the training data, for example:

```python
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
```

and make separate dataloaders for the validation and training datasets:

```python

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, sampler=valid_sampler, num_workers=2)
```

(don't forget to define your trainset at the  start)
