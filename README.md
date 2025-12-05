# Concrete Strength Predictor

A web application for predicting concrete compressive strength based on region, mortar type, and rock ratios.

## Prerequisites

This project uses [Bun](https://bun.sh) as the package manager and runtime. The project includes a bundled Bun binary in the `bin/` directory (which runs with `docker`), so you don't need to install Bun separately.

## Installation

*The steps below are meant to be run from the `frontend/` directory.*

```bash
cd frontend
```


Install project dependencies:

```bash
make install
```

This will run `./bin/bun install --frozen-lockfile` to install all dependencies.

## Development

Start the development server:

```bash
make dev
```

This starts the Vite development server. The application will be available at `http://localhost:5173` (or the next available port).

## Building

Build the production bundle (this is a good test to run before pushing to continuous deployment):

```bash
make build
```

This runs TypeScript type checking and builds the optimized production bundle to the `dist/` directory.

## Linting

Run ESLint to check and fix code style issues:

```bash
make lint
```

## Docker

### Build Docker Image

Build a Docker image for the application:

```bash
make docker-build
```

This creates a Docker image tagged as `mindthemath/concrete-app`.

### Run Docker Container

Run the application in a Docker container:

```bash
make docker-run
```

This starts the container and maps port 3000 from the container to port 3030 on your host machine. Access the application at `http://localhost:3030`.

## Cleaning

Remove the build output directory:

```bash
make clean
```

This deletes the `dist/` directory.

## Available Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install project dependencies |
| `make dev` | Start development server |
| `make build` | Build production bundle |
| `make lint` | Run ESLint |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make clean` | Remove build artifacts |

## Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Bun** - Package manager and runtime
- **Tailwind CSS** - Styling
- **ESLint** - Code linting

## Project Structure

```
frontend/
├── bin/              # Bundled Bun binary
├── src/              # Source code
│   ├── components/   # React components
│   ├── themes/       # Theme definitions
│   └── ...
├── data/             # Data files (regions, concrete, etc.)
├── dist/             # Build output (generated)
└── makefile          # Build commands
```

## Backend

[To be added]
- [ ] add `uv` and `uvx` binaries to the `bin/` directory inside of `backend/`
- [ ] document litserve and slowapi usage, building docker container, etc.
- [ ] `VITE_API_ENDPOINT` needs to be updated in GHA when hosted endpoint changes. Same idea for the domain of the site.
- [ ] note that the endpoint will be publicly exposed on the frontend, so it's not sensitive info (no sensitivity of secrets in GHA def nor the .env file).
