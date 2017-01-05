# Build
```
docker build -t cannin/jupyter-keras-tensorflow-tools:tf-0.12.1-py3 .
docker build -t cannin/jupyter-keras-tensorflow-tools-sshd:tf-0.12.1-py3 -f Dockerfile_ssh .
```

# SSH
```
docker rm -f sshd; docker run -d --name sshd -p 23:22 -v $(pwd):/keras -w /keras -t cannin/jupyter-keras-tensorflow-tools-sshd
ssh -p 23 root@localhost
```
