# Linux AI NPU Helper

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://robertbiv.github.io/Linux-AI-NPU-Helper/)
[![Tests](https://github.com/robertbiv/Linux-AI-NPU-Helper/actions/workflows/docs.yml/badge.svg)](https://github.com/robertbiv/Linux-AI-NPU-Helper/actions/workflows/docs.yml)
[![Coverage](https://robertbiv.github.io/Linux-AI-NPU-Helper/coverage-badge.json)](https://robertbiv.github.io/Linux-AI-NPU-Helper/coverage/)

A privacy-first AI assistant for Linux that runs entirely on the **AMD Ryzen AI NPU** — no cloud, no telemetry, no API keys required.

📖 **[Full documentation → robertbiv.github.io/Linux-AI-NPU-Helper](https://robertbiv.github.io/Linux-AI-NPU-Helper/)**

## Features

- 🧠 **NPU-accelerated inference** via ONNX Runtime + VitisAI (AMD Ryzen AI)
- 🦙 **Ollama & OpenAI-compatible backends** (LM Studio, Jan, etc.)
- 🔒 **100% local** — all data stays on your machine
- 🖥️ **Desktop-native UI** — automatically matches GNOME, KDE, XFCE, MATE, Cinnamon, Pantheon, Deepin, tiling WMs
- 🛠️ **10 built-in tools** — file search, web fetch, man pages, system control, app/process management
- ⚙️ **GUI settings page** — all settings sync instantly to JSON
- 🩺 **Diagnostic menu** — live status of every subsystem + integrated test runner
- 🔑 **Copilot key support** — AMD Ryzen AI laptops, ASUS, Lenovo

## Quick start

```bash
git clone https://github.com/robertbiv/Linux-AI-NPU-Helper.git
cd Linux-AI-NPU-Helper
pip install -r requirements.txt

# Start Ollama and pull a model
ollama serve &
ollama pull llama3.2:3b-instruct-q4_K_M

python -m src
```

## Documentation

- [Building the Flatpak locally](https://robertbiv.github.io/Linux-AI-NPU-Helper/guides/building-flatpak/)
- [AI Model Guide — browse, add, delete models](https://robertbiv.github.io/Linux-AI-NPU-Helper/guides/model-guide/)
- [Settings Guide](https://robertbiv.github.io/Linux-AI-NPU-Helper/guides/settings-guide/)
- [API Reference](https://robertbiv.github.io/Linux-AI-NPU-Helper/api/config/)

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src
```
