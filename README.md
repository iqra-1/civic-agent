# 🏛️ LocalGov AI Agent

An **open-source, agentic AI system** that answers citizen queries about Dublin City Council services using **retrieval-augmented generation (RAG)** and **multi-agent reasoning** — all powered by **free, local LLMs** (Mistral-7B).

> 🔍 Example: *"I'm a student in Harold's Cross — can I get a bin collection exemption?"*  
> ✅ AI responds with: eligibility rules + next steps + official links.

## ✨ Features
- **Multi-agent simulation**: Researcher → Validator → Actioner
- **Real-time RAG** on scraped Dublin City Council policies
- **Runs entirely offline** using quantized LLMs (no API costs)
- **Gradio UI** for easy demo & testing
- **GPU-accelerated** via `llama-cpp-python`

## 🛠️ Tech Stack
- **LLM**: Mistral-7B-Instruct (GGUF, Q5 quantized)
- **Agents**: Simulated CrewAI-style workflow
- **RAG**: FAISS + `all-MiniLM-L6-v2` embeddings
- **Backend**: LangChain + local LLM
- **UI**: Gradio
- **Deployment-ready**: Easily containerized for AWS/Azure
## ▶️ Try It
1. Open [`localgov_ai_agent.ipynb`](./localgov_ai_agent.ipynb) in Google Colab  
2. **Runtime → Change runtime type → GPU**  
3. Run all cells  
4. At the end, a public Gradio link appears — test live!


## 📦 Dependencies
See notebook install block. Uses only open-source, free tools.

## 🚀 Future Work
- Integrate **CrewAI** for true agent delegation
- Add **n8n** for email/SMS follow-ups
- Deploy on **Azure Container Apps** or **AWS EC2**

---

Built with ❤️ by Iqra  
*Part of my journey to master agentic AI, LLMs, and real-world problem solving.*
