[Original README](https://github.com/chainer/chainercv/tree/master/examples/ssd)

# Modified Examples of Single Shot Multibox Detector

## Train
You can train the model with the following code.
For additional params look at the [Original README](https://github.com/chainer/chainercv/tree/master/examples/ssd).

```
$ python train.py --imgs <path_to_labeled_images>
```

**Assumptions**:
* The labeled_images folder contains only *.jpg and *.xml files.
* Every image in the folder has been checked for buildings.
* If a .xml file is not present for a given picture, the picture has no bounding boxes.
