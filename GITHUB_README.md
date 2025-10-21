# GigaBase - Advanced LLM System with Enhanced Example-Based Generation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An advanced Language Model system featuring **Model-Analyzed, Dataset-Only Generation** - where AI intelligence meets zero-hallucination accuracy.

## 🌟 Key Features

### 🎯 Three Generation Modes

1. **Example-Only Mode (Default)** ⭐ RECOMMENDED
   - Model analyzes query intent and extracts key concepts
   - Multi-strategy search (3 parallel strategies)
   - Output comes 100% from dataset (zero hallucination)
   - Ultra-fast search: 3-30 microseconds
   - Response time: 7-20 seconds

2. **Hybrid Mode** (Model + Dataset)
   - RAG (Retrieval Augmented Generation)
   - Combines model intelligence with dataset grounding
   - Factual outputs with creative understanding

3. **Model-Only Mode** (Fallback)
   - Pure model generation
   - Deterministic mode for consistency

### ⚡ Performance

- **Search Speed**: 3-30 microseconds (ultra-fast indexing)
- **Multi-level Indexing**: Keywords, bigrams, trigrams
- **LRU Cache**: 1000 queries cached
- **Response Time**: 7-20 seconds (with model analysis)
- **Confidence Scoring**: High/Medium/Low quality indicators

### 🧠 Smart Features

- **Intent Detection**: Automatically detects question type (definition/howto/example/explanation)
- **Concept Extraction**: Identifies key concepts from queries
- **Multi-Strategy Search**: 3 parallel search strategies for best results
- **Smart Ranking**: Combines scores from multiple strategies
- **Duplicate Removal**: Ensures unique, high-quality content
- **Source Tracking**: Shows which dataset files contributed to answers

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GigaBase.git
cd GigaBase

# Install dependencies
pip install -r requirements.txt

# Run the server
python run_purestock_server.py
```

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           Model Analysis (Intent + Concepts)                 │
│  - Question type detection                                   │
│  - Key concept extraction                                    │
│  - Answer format determination                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-Strategy Search (3 Strategies)                 │
│  1. Direct query search (highest weight)                     │
│  2. Concept-based search                                     │
│  3. Intent-based keyword expansion                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Smart Ranking & Filtering                            │
│  - Combine scores from all strategies                        │
│  - Boost multi-strategy matches                              │
│  - Intent-aware sentence scoring                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Refined Extraction                                   │
│  - Score sentences (query + concept + intent)                │
│  - Remove duplicates                                         │
│  - Optimize sentence count                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         100% Dataset-Only Output                             │
│         (Zero Hallucination)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
GigaBase/
├── app/
│   ├── app.py                          # Flask web server
│   ├── model_utils.py                  # Model loading utilities
│   ├── fast_search.py                  # Ultra-fast search engine
│   ├── example_based_generation.py     # Enhanced example-based generator
│   ├── hybrid_generation.py            # Hybrid RAG generator
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── templates/
│       └── index.html
├── data/
│   ├── raw/                            # Raw dataset files
│   └── processed/                      # Processed datasets
├── models/
│   ├── pretrained/                     # Pretrained models
│   └── fine_tuned/                     # Fine-tuned models
├── docs/
│   ├── EXAMPLE_BASED_GUIDE.md         # Example-based generation guide
│   ├── HYBRID_SYSTEM_GUIDE.md         # Hybrid system guide
│   └── TRAINING_SPEED_GUIDE.md        # Training optimization guide
├── run_purestock_server.py            # Optimized server runner
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🎮 Usage Examples

### Example 1: Definition Query

**Input**: "What is machine learning?"

**Process**:
1. Model detects intent: `definition`
2. Extracts concepts: `[machine, learning]`
3. Multi-strategy search finds 5 examples
4. Extracts relevant sentences from dataset
5. Combines top examples

**Output**: Dataset-only answer with confidence score

### Example 2: How-to Query

**Input**: "How to train a neural network?"

**Process**:
1. Intent: `howto`
2. Concepts: `[train, neural, network]`
3. Searches for step-by-step content
4. Prioritizes sentences with process indicators

**Output**: Step-by-step answer from dataset examples

## 🔧 Configuration

### Generation Modes

Edit `app/static/js/main.js` to change default mode:

```javascript
// Example-only mode (default)
"mode": "example-only"

// Hybrid mode
"mode": "hybrid"

// Model-only mode
"mode": "model-only"
```

### Search Parameters

Edit `app/example_based_generation.py`:

```python
# Number of examples to search
num_examples=5

# Maximum output length
max_length=400
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Search Speed | 3-30 μs |
| Total Response Time | 7-20 seconds |
| Model Parameters | 124M |
| Keyword Index Size | 16,359 keywords |
| N-gram Index Size | 142,641 n-grams |
| Dataset Files | 5 files |
| Hallucination Rate | 0% (dataset-only) |

## 🛠️ API Endpoints

### POST `/generate`

Generate text from prompt

**Request Body**:
```json
{
  "model_id": "Purestock",
  "prompt": "What is machine learning?",
  "max_tokens": 150,
  "temperature": 0.0,
  "deterministic": true,
  "mode": "example-only"
}
```

**Response**:
```json
{
  "prompt": "What is machine learning?",
  "generated_texts": ["Dataset-based answer..."],
  "token_count": 113,
  "word_count": 84,
  "generation_time": 7.8,
  "mode": "example-only",
  "quality": "dataset-accurate",
  "metadata": {
    "intent": "definition",
    "answer_type": "detailed",
    "confidence": "high",
    "sources": ["ai_ml_dl", "stackexchange"],
    "top_scores": [64.67, 53.67, 43.67]
  }
}
```

## 📚 Documentation

- [Example-Based Generation Guide](docs/EXAMPLE_BASED_GUIDE.md)
- [Hybrid System Guide](docs/HYBRID_SYSTEM_GUIDE.md)
- [Training Speed Optimization](docs/TRAINING_SPEED_GUIDE.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Flask web framework
- distilgpt2 base model
- Open-source community

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for accurate, hallucination-free AI responses**
