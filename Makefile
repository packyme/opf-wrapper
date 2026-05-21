PYTHON ?= .venv/bin/python
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help run download-model benchmark stress

help:
	@echo "Available targets:"
	@echo "  make download-model  Download the ONNX q4 model into ./model"
	@echo "  make run             Start the FastAPI server"
	@echo "  make benchmark       Benchmark the running HTTP server"
	@echo "  make benchmark ARGS=\"--text-mode long --long-chars 120000 --requests 5\""
	@echo "  make stress          Stress test the running HTTP server"
	@echo "  make stress ARGS=\"--requests 1000 --concurrency 50 --text-mode long\""

run:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

download-model:
	$(PYTHON) -m scripts.download_model

benchmark:
	$(PYTHON) -m scripts.benchmark $(ARGS)

stress:
	$(PYTHON) -m scripts.benchmark --requests 500 --warmup 20 --concurrency 20 --text-mode mixed --long-chars 65536 $(ARGS)
