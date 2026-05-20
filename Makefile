PYTHON ?= .venv/bin/python
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help run download-model benchmark

help:
	@echo "Available targets:"
	@echo "  make download-model  Download the ONNX q4 model into ./model"
	@echo "  make run             Start the FastAPI server"
	@echo "  make benchmark       Benchmark the running HTTP server"
	@echo "  make benchmark ARGS=\"--text-mode long --long-chars 120000 --requests 5\""

run:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

download-model:
	$(PYTHON) -m scripts.download_model

benchmark:
	$(PYTHON) -m scripts.benchmark $(ARGS)
