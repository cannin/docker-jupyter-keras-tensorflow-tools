# Build
```
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.0.1 .
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-1.0.1-gpu -f Dockerfile.gpu .

docker build -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-1.0.1-py3 -f Dockerfile_ssh .
```

# Run
```
docker rm -f keras; docker run --name keras -v DIR:/notebooks -p 8888:8888 -t cannin/jupyter-keras-tensorflow-tools:tf-1.0.1

docker exec -i -t keras bash
```

# SSH
```
docker rm -f sshd; docker run -d --name sshd -p 23:22 -p 8888:8888 -v $(pwd):/notebooks -w /notebooks -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-0.12.1-py3
docker rm -f sshd; docker run --name sshd -p 23:22 -p 8888:8888 -v $(pwd):/notebooks -w /notebooks -it cannin/jupyter-keras-tensorflow-tools-sshd:tf-0.12.1-py3 bash
docker exec -it sshd bash
ssh -p 23 root@localhost
```
