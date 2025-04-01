# Test

## Features

- Feature 1: Description of the first key feature
- Feature 2: Description of the second key feature
- Feature 3: Description of the third key feature
- [Add more features as needed]

## Prerequisites

- Docker installed on your system
- [Any other prerequisites]

## Getting Started

### Running with Docker

The easiest way to run this application is using the provided Docker script:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Run the application using one of these methods:

   **Option 1**: Make the script executable first (recommended):
   ```bash
   chmod +x docker_run.sh
   ./docker_run.sh
   ```

   **Option 2**: Run with bash directly (no need to change permissions):
   ```bash
   bash docker_run.sh
   ```

   **Option 3**: Use source (no need to change permissions):
   ```bash
   source docker_run.sh
   ```

4. Install llm2vec

```bash
cd /data/research/model/llm2vec/llm2vec
pip install -e .
```
5. Start conda

```bash
source activate llm
```


### Configuration Options

The `docker_run.sh` script accepts the following parameters:

- `--param1=value`: Description of parameter 1
- `--param2=value`: Description of parameter 2
- [Add more parameters as needed]

Example:
```

## Troubleshooting

- **Permission denied error**: If you see `-bash: ./docker_run.sh: Permission denied`, you have three options:
  1. Make the script executable: `chmod +x docker_run.sh`
  2. Run with bash directly: `bash docker_run.sh`
  3. Use source: `source docker_run.sh`
  