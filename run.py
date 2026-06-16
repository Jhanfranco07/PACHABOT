from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the FastAPI application."""

    print("Iniciando el servidor de PachaBot...")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
