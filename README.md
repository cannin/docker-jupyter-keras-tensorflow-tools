# Build
```
docker build -t cannin/jupyter-keras-tensorflow-tools .
docker build -t cannin/jupyter-keras-tensorflow-tools-sshd -f Dockerfile_ssh .
```

# SSH
```
docker rm -f sshd; docker run -d --name sshd -p 23:22 -v $(pwd):/keras -w /keras -t cannin/jupyter-keras-tensorflow-tools-sshd
ssh -p 23 root@localhost
```
