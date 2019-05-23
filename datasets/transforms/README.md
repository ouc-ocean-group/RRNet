# Transforms
All the input data in these transforms is a tuple consists of the image and the annotations.

- HorizontalFlip
- ToTensor
- Normalize
- RandomCrop
- ColorJitter
- MaskIgnore

The recommended transforms order is:

```python
ColorJitter > ToTensor > HorizontalFlip > RandomCrop > Normalize > MaskIgnore
```

A transform demo:

```python
img = Image.open('./data/demo/images/0000364_01765_d_0000782.jpg')
with open('./data/demo/annotations/0000364_01765_d_0000782.txt', 'r') as reader:
    annos = reader.readlines()
test_data = (img, annos)

trans = Compose([
    ToTensor(),
    HorizontalFlip(),
    RandomCrop((500, 500))
])

test_data = trans(test_data)

img = test_data[0]  # a tensor, c * h * w
annos = test_data[1]  # a tensor, n * (y, x, h, w, score, object_category, truncation, occlusion)

```

