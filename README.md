# Build
```
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 .
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-gpu-py3 -f Dockerfile_gpu .

docker build -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 -f Dockerfile_ssh .
```

# Run
```
## Jupyter
docker rm -f keras; docker run --name keras -v $(pwd):/notebooks -p 8888:8888 -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3

docker rm -f keras; docker run --name keras -v $(pwd):/notebooks -p 8888:8888 -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 jupyter notebook --allow-root --no-browser

## Bash
docker rm -f keras; docker run --name keras -it cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 bash

## Interactive shell
docker exec -i -t keras bash
```

# SSH
```
docker rm -f sshd; docker run -d --name sshd -p 23:22 -p 8888:8888 -v $(pwd):/notebooks -w /notebooks -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3

docker rm -f sshd; docker run --name sshd -p 23:22 -p 8888:8888 -v $(pwd):/notebooks -w /notebooks -it cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 bash

docker rm -f sshd; docker run --name sshd -p 23:22 -p 8888:8888 -v $(pwd):/notebooks -w /notebooks -it cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 jupyter lab --allow-root --no-browser

docker exec -it sshd bash
ssh -p 23 root@localhost
```

# Check Versions
```
import cv2
import tensorflow as tf
import keras

# Get versions
print(cv2. __version__)
print(tf.__version__)
print(keras.__version__)

# Test GPU
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
```
