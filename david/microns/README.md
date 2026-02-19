# Build the Docker Image

In the current directory, run the `docker build` command

```bash
$ cd <project_root>/david/microns
$ docker build -t davidkopala/microns-downsample -f package/Dockerfile .
$ docker run davidkopala/microns-downsample
```

## Executing
```bash
$ mkdir output
$ docker run --rm \
    -v ./output:/output \
    -v ~/.cloudvolume:/root/.cloudvolume \
    davidkopala/microns-downsample \
    python app.py -s /output 0 0 0
```

```powershell
PS> docker run --rm `
    -v ./output:/output `
    -v $HOME/.cloudvolume:/root/.cloudvolume `
    davidkopala/microns-downsample `
    python app.py -s /output 0 0 0
```
