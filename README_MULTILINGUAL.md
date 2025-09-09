# 🌍 Chatterbox TTS Server - Multilingual Edition

Advanced Text-to-Speech server with multilingual support based on Chatterbox TTS models.

## ✨ Features

- 🎯 **Dual Model Support**: English ChatterboxTTS + Multilingual ChatterboxMultilingualTTS
- 🌍 **16 Languages**: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean
- 🚀 **Smart Model Selection**: Automatically chooses optimal model based on language
- 🎨 **Modern Web UI**: Clean, responsive interface with dark/light themes
- 🐳 **Docker Ready**: Full containerization with GPU support
- ⚡ **GPU Optimized**: NVIDIA CUDA support for fast inference
- 🔧 **FastAPI Backend**: High-performance async API
- 📊 **Voice Cloning**: Upload reference audio for voice matching

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/chatterbox-multilingual-server.git
cd chatterbox-multilingual-server

# Run with GPU support
docker-compose up

# Or build manually
docker build -t chatterbox-multilingual .
docker run -p 8004:8004 --gpus all chatterbox-multilingual
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements-nvidia.txt  # For GPU
# or
pip install -r requirements.txt         # For CPU

# Run server
python server.py
```

## 🌐 Supported Languages

| Code | Language | Model |
|------|----------|-------|
| `en` | English | ChatterboxTTS (Optimized) |
| `es` | Spanish | ChatterboxMultilingualTTS |
| `fr` | French | ChatterboxMultilingualTTS |
| `de` | German | ChatterboxMultilingualTTS |
| `it` | Italian | ChatterboxMultilingualTTS |
| `pt` | Portuguese | ChatterboxMultilingualTTS |
| `pl` | Polish | ChatterboxMultilingualTTS |
| `tr` | Turkish | ChatterboxMultilingualTTS |
| `ru` | Russian | ChatterboxMultilingualTTS |
| `nl` | Dutch | ChatterboxMultilingualTTS |
| `cs` | Czech | ChatterboxMultilingualTTS |
| `ar` | Arabic | ChatterboxMultilingualTTS |
| `zh` | Chinese | ChatterboxMultilingualTTS |
| `ja` | Japanese | ChatterboxMultilingualTTS |
| `hu` | Hungarian | ChatterboxMultilingualTTS |
| `ko` | Korean | ChatterboxMultilingualTTS |

## 📡 API Usage

### Generate Speech

```bash
curl -X POST "http://localhost:8004/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a multilingual TTS test!",
    "language": "en",
    "temperature": 0.8,
    "exaggeration": 1.3
  }'
```

### Supported Parameters

- `text`: Text to synthesize
- `language`: Language code (default: "en")
- `temperature`: Randomness (0.1-2.0, default: 0.8)
- `exaggeration`: Expressiveness (0.1-3.0, default: 1.3)
- `cfg_weight`: Guidance weight (0.1-1.0, default: 0.5)
- `seed`: Random seed for reproducibility
- `voice_mode`: "predefined" or "reference"
- `voice_id`: Voice file name (for predefined voices)

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
server:
  host: 0.0.0.0
  port: 8004

tts_engine:
  device: cuda  # or cpu, auto

generation_defaults:
  temperature: 0.8
  exaggeration: 1.3
  language: en

ui:
  title: "Chatterbox TTS Server (Multilingual)"
  show_language_select: true
```

## 🐳 Docker Images

Pre-built images available on GitHub Container Registry:

```bash
# Latest stable
docker pull ghcr.io/YOUR_USERNAME/chatterbox-multilingual-server:latest

# Specific version
docker pull ghcr.io/YOUR_USERNAME/chatterbox-multilingual-server:main
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│   Web UI        │───▶│   FastAPI       │───▶│   Engine Manager     │
│   (Multi-lang)  │    │   Server        │    │   (Model Selector)   │
└─────────────────┘    └─────────────────┘    └──────────┬───────────┘
                                                         │
                              ┌──────────────────────────┼──────────────────────────┐
                              │                          │                          │
                              ▼                          ▼                          │
                    ┌─────────────────┐        ┌─────────────────────┐              │
                    │ ChatterboxTTS   │        │ChatterboxMultilingual│              │
                    │   (English)     │        │    TTS (23 langs)    │              │
                    └─────────────────┘        └─────────────────────┘              │
                              │                          │                          │
                              └──────────────────────────┼──────────────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │   Audio Output  │
                                              │  (WAV/MP3/etc)  │
                                              └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Resemble AI](https://github.com/resemble-ai/chatterbox) for the original Chatterbox TTS models
- [devnen](https://github.com/devnen/chatterbox) for the stable server architecture
- Community contributors for multilingual enhancements

## 🔗 Links

- [Original Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [Hugging Face Models](https://huggingface.co/ResembleAI/chatterbox)
- [Docker Hub](https://hub.docker.com)
- [Documentation](./documentation.md)
