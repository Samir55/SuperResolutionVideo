# Graph Scripts

## To Convert Keras Model to TensorFlow Graph .pb (Protobuf) file
First copy the .h5 pretrained model to graph_scripts directory, then run

```Console

python3 keras_to_tensorflow.py  --input_model="keras_model.h5"  --output_model="model.pb"

```

## To Show The Graph at tensorboard
```Console

python3 import_pb_to_tensorboard.py --model_dir="model.pb" --log_dir="tensorboard_dir/"

```

## To Run tensorborad
```Console

tensorboard --logdir tensorboard_dir/

```
where you can see the graph at the link provided.