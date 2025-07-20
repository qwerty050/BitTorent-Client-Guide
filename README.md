# A Simplified BitTorrent Client in Python

This repository contains the skeleton code for building a simplified BitTorrent client using Python and `asyncio`.

The goal is to fill in the function and class definitions in the `pieces/` directory. The test suite in the `tests/` directory is already complete. As you implement the functionality, the tests will start to pass.

Follow guide.md for instructions.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd bittorrent-client
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv bittorrent
    source bittorrent/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Tests

To check your implementation, run the test suite using `pytest`:

```bash
pytest -v
