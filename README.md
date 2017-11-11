# Build
```
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 .
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-gpu-py3 -f Dockerfile_gpu .

docker build -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 -f Dockerfile_ssh .
```

# Run
```
## Jupyter
docker rm -f keras; docker run --name keras -v $(pwd):/notebooks -p 8888:8888 -p 6006:6006 -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3

docker rm -f keras; docker run --name keras -v $(pwd):/notebooks -p 8888:8888 -p 6006:6006 -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 jupyter lab --allow-root --no-browser

## Bash
docker rm -f keras; docker run --name keras -i -v $(pwd):/notebooks -p 8888:8888 -p 6006:6006 -t cannin/jupyter-keras-tensorflow-tools:tf-1.4.0-devel-py3 bash

## Interactive shell
docker exec -i -t keras bash
```

# SSH
```
docker rm -f keras; docker run -d --name keras -p 23:22 -p 8888:8888 -p 6006:6006 -v $(pwd):/notebooks -w /notebooks -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3

docker rm -f keras; docker run --name keras -p 23:22 -p 8888:8888 -p 6006:6006 -v $(pwd):/notebooks -w /notebooks -it cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 bash

docker rm -f keras; docker run --name keras -p 23:22 -p 8888:8888 -p 6006:6006 -v $(pwd):/notebooks -w /notebooks -it cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.4.0-devel-py3 jupyter lab --allow-root --no-browser

docker exec -it sshd bash
ssh -p 23 root@localhost
```
