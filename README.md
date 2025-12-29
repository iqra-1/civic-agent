# 🏛️ Dublin City Council AI Assistant

A multi-agent RAG (Retrieval-Augmented Generation) system that provides intelligent responses to queries about Dublin City Council policies, services, and procedures using local LLMs.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🎯 Overview

This project implements a sophisticated AI assistant that combines:
- **Document Retrieval (RAG)**: Searches through official Dublin City Council documents
- **Multi-Agent System**: Three specialized AI agents work together
- **Local LLM**: Privacy-focused, GPU-accelerated inference using Ollama + Phi-3
- **Web Interface**: User-friendly Gradio chat interface

### Why This Architecture?

**1. RAG (Retrieval-Augmented Generation)**
- ✅ Grounds responses in official documents (no hallucinations)
- ✅ Always up-to-date with latest policies
- ✅ Provides source citations

**2. Multi-Agent System (CrewAI)**
- ✅ Specialized agents = better quality
- ✅ Separation of concerns (research → validate → advise)
- ✅ More reliable than single-agent systems

**3. Local LLM (Ollama + Phi-3)**
- ✅ Privacy: No data sent to external APIs
- ✅ Cost: Free inference, no API costs
- ✅ Speed: GPU acceleration on T4/A100
- ✅ Phi-3: Microsoft's efficient 3.8B parameter model


[//]: # ("Comment")
[Comment test]::
[//]: # (This may be the most platform independent comment)
## 🏗️ System Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│   CrewAI Multi-Agent System         │
│                                     │
│   Agent 1: Policy Researcher        │
│   ├─ Uses RAG tool                  │
│   └─ Searches FAISS vector DB       │
│                                     │
│   Agent 2: Eligibility Validator    │
│   └─ Interprets policies strictly   │
│                                     │
│   Agent 3: Citizen Action Guide     │
│   └─ Provides actionable steps      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   FAISS Vector Database             │
│   (Dublin City Council docs)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Ollama + Phi-3 (Local LLM)        │
│   (GPU-accelerated inference)       │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended for speed)
- 8GB+ RAM
- Google Colab (or local setup)
- Internet connection (for web scraping)

### Installation

```bash
# 1. Install dependencies
pip install --upgrade \
    "numpy>=2.0,<2.3" \
    "scipy>=1.13.0" \
    "transformers" \
    "sentence-transformers" \
    "faiss-cpu" \
    "langchain" \
    "langchain-community" \
    "crewai" \
    "crewai-tools" \
    "gradio" \
    "beautifulsoup4" \
    "requests"

# 2. Install and setup Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5
ollama pull phi3

# 3. Run the notebook cells in order
```

### Usage

1. **Prepare Documents**: Place Dublin City Council PDFs/documents in `/content/data/`
2. **Build FAISS Index**: Run the document processing cells
3. **Run Queries**: Use the CLI interface

```python
# Single query
single_query_mode("Your question here")

# Interactive mode
interactive_mode()

# Batch queries
for query in my_queries:
    single_query_mode(query)
```

## 📁 Project Structure

```
dublin-council-ai/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── dublin_council_rag.ipynb  # Main Colab notebook
├── src/
│   ├── scraper.py                 # Web scraping module
│   ├── document_processor.py     # Text → FAISS pipeline
│   ├── agents.py                  # CrewAI agent definitions
│   ├── rag_tool.py                # Custom RAG tool
│   └── cli.py                     # Command-line interface
├── data/
│   ├── raw/                       # Scraped text + optional PDFs
│   └── faiss_index/               # Generated vector database
└── models/
    └── phi-3-mini-q4.gguf         # Downloaded LLM (not in git)
```

## 🔧 Technical Deep Dive

### Why Each Component?

#### 1. **FAISS Vector Database**
**Chosen over**: Pinecone, Weaviate, ChromaDB

**Reasons**:
- ✅ Runs locally (no external dependencies)
- ✅ Extremely fast similarity search
- ✅ Low memory footprint
- ✅ Facebook Research's battle-tested library
- ✅ Works offline

#### 2. **Sentence Transformers (all-MiniLM-L6-v2)**
**Chosen over**: OpenAI embeddings, large BERT models

**Reasons**:
- ✅ Only 80MB model size
- ✅ Fast inference (384-dim embeddings)
- ✅ Excellent quality for semantic search
- ✅ Free and runs locally
- ✅ Widely used and trusted

#### 3. **CrewAI Multi-Agent Framework**
**Chosen over**: LangChain agents, AutoGPT, single LLM

**Reasons**:
- ✅ Purpose-built for multi-agent workflows
- ✅ Clean agent definitions with roles/goals
- ✅ Built-in task orchestration
- ✅ Easy tool integration
- ✅ Better than single agent for complex queries

#### 4. **Ollama + Phi-3**
**Chosen over**: API models (GPT-4, Claude), other local LLMs

**Reasons**:
- ✅ **Ollama**: Easy setup, automatic GPU offloading
- ✅ **Phi-3**: 3.8B params, optimized for efficiency
- ✅ Matches GPT-3.5 quality at 1/50th the size
- ✅ Works with CrewAI's LiteLLM backend
- ✅ Microsoft-backed, well-maintained

**Why not llama-cpp directly?**
- ❌ CrewAI's newer versions use LiteLLM
- ❌ LiteLLM doesn't recognize raw llama-cpp objects
- ✅ Ollama provides the compatibility layer

<!--  #### 5. **Command Line Interface**
[//]: <>  **Chosen over**: Gradio, Streamlit, Flask

[//]: <>  **Reasons**:
[//]: <>  - ✅ Works in any environment (Colab, Jupyter, local)
[//]: <>  - ✅ No port/threading issues
[//]: <>  - ✅ Easy to script and automate
[//]: <>  - ✅ Perfect for batch processing
[//]: <>  - ✅ Simple to understand and modify
[//]: <>  - ✅ Saves conversation history automatically-->

### Key Design Decisions

**1. Sequential Agent Processing**
```python
Researcher → Validator → Actioner
```
- Ensures information flows logically
- Each agent builds on previous output
- More reliable than parallel processing

**2. Limited RAG Retrieval (k=3)**
- Prevents context overflow
- Faster processing
- Forces focus on most relevant docs

**3. Low Temperature (0.1)**
- Deterministic, consistent responses
- Less creative but more factual
- Critical for policy interpretation

**4. Max Iterations = 3**
- Allows tool usage but prevents loops
- Balances thoroughness vs. speed

## 🎓 Step-by-Step Explanation

### Step 1: Document Processing
```python
# Why: Get latest information from official website
WebScraper → BeautifulSoup → Clean Text → FAISS
```
- **WebScraper**: Fetches 15+ key service pages
- **BeautifulSoup**: Extracts clean text, removes navigation
- **Optional PDFs**: Can add policy documents for more coverage
- **FAISS**: Stores vectors for fast similarity search

<!-- **Why web scraping over PDF upload?**
[//]: <>  - ✅ Always up-to-date information
[//]: <>  - ✅ No manual document collection
[//]: <>  - ✅ Covers breadth of services
[//]: <>  - ✅ Easy to add new pages
[//]: <>  - ⚠️ PDFs still supported as supplement-->

### Step 2: RAG Tool Creation
```python
# Why: Bridge between agents and documents
class DublinCouncilRAGTool(BaseTool):
    def _run(self, query: str) -> str:
        docs = retriever.invoke(query)  # Semantic search
        return formatted_excerpts
```
- CrewAI agents can call this tool
- Returns top 3 most relevant excerpts
- Truncates to 300 chars to fit context

### Step 3: Agent Definitions
```python
# Why: Specialized roles improve quality

# Researcher: Only searches, never guesses
researcher = Agent(
    llm=ollama_llm,
    tools=[rag_tool],  # Has access to documents
    max_iter=3         # Can search multiple times
)

# Validator: Strict YES/NO decisions
validator = Agent(
    llm=ollama_llm,
    tools=[],          # No tools = only analyzes
)

# Actioner: Practical guidance
actioner = Agent(
    llm=ollama_llm,
    tools=[]           # Uses validator's decision
)
```

### Step 4: Task Orchestration
```python
# Why: Clear information flow

task1 = Task(
    description="Search for policy info",
    agent=researcher,
    context=[]  # No dependencies
)

task2 = Task(
    description="Determine eligibility",
    agent=validator,
    context=[task1]  # Sees researcher's output
)

task3 = Task(
    description="Provide next steps",
    agent=actioner,
    context=[task1, task2]  # Sees both outputs
)
```
<!--  ### Step 5: Gradio Interface
[//]: <>  ```python
[//]: <>  # Why: User-friendly chat interface

[//]: <>  def process_query(query, history):
[//]: <>      result = crew.kickoff()  # Run agents
[//]: <>      return formatted_response

[//]: <>  demo = gr.ChatInterface(
[//]: <>      fn=process_query,
[//]: <>      examples=[...],  # Suggested queries
[//]: <>      share=True       # Public URL
[//]: <>  )
[//]: <>  ```-->

## 📊 Performance

| Metric | Value |
|--------|-------|
| Query Response Time | 10-30 seconds |
| Document Retrieval | < 1 second |
| LLM Inference | 8-25 seconds |
| GPU Memory Usage | ~4GB |
| Accuracy (subjective) | ~85% |

## 🔒 Privacy & Security

- ✅ All processing happens locally
- ✅ No data sent to external APIs
- ✅ Documents stay on your server
- ✅ GDPR compliant
- ⚠️ Still verify critical information with official sources

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'transformers.modeling_layers'"
```bash
pip install --upgrade transformers accelerate
```

### "BadRequestError: LLM Provider NOT provided"
```bash
# Ensure Ollama is running
ollama serve &
ollama list  # Should show phi3
```

### "NumPy/SciPy version conflicts"
```bash
pip uninstall -y numpy scipy
pip install "numpy>=2.0,<2.3" "scipy>=1.13.0"
# Restart runtime
```

### FAISS index not loading
```bash
# Rebuild the index
vectorstore.save_local("/content/data/faiss_index")
```

## 🚧 Future Improvements

- [ ] Add conversation memory (multi-turn)
- [ ] Implement streaming responses
- [ ] Add document upload via Gradio
- [ ] Multi-language support (Irish + English)
- [ ] Voice input/output
- [ ] Mobile app version
- [ ] User feedback collection
- [ ] A/B testing different LLMs
- [ ] Automated policy updates

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dublin City Council for public documents
- Anthropic for Claude (used in development)
- Microsoft for Phi-3 model
- Ollama team for inference server
- CrewAI for multi-agent framework
- HuggingFace for embeddings

---

**Built with ❤️ for Dublin citizens**
