# api definition

The `Dockerfile` here is built from context two directories up (see `makefile`), and is meant to serve as a reference point for an indepdent backend image.
It's primary use-case is testing the application backend independently; the `Dockerfile` in the root directory is the final production image which bundles in both the API and frontend.


For local development, `uv` is used with `make dev` to run an instance of the API without the use of docker. This is completely optional - rebuilding the image is a fine way to test code changes as well.

To prepare the final production bundle, `make data` will `cd` into the `../scripts` directory and compress the artifacts required for inclusion in the final image.
`make clean` will remove the `artifacts.tar.gz` file created by `make data`.

