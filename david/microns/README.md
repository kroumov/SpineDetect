# Build the Docker Image

In the current directory, run the `docker build` command

```bash
$ cd <project_root>/david/microns
$ docker build -t downsample -f package/Dockerfile .
$ docker run downsample
```

## Executing
```bash
$ mkdir output
$ docker run --rm \
    -v ./output:/output \
    -v ~/.cloudvolume:/root/.cloudvolume \
    downsample \
    python app.py -s /output 0 0 0
```

```powershell
PS> docker run --rm `
    -v ./output:/output `
    -v $HOME/.cloudvolume:/root/.cloudvolume `
    downsample `
    python app.py -s /output 0 0 0
```
