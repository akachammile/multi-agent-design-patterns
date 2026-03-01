# Using Claude Code with This Project

This project is designed to be Claude Code friendly. This guide helps you (and Claude) work effectively with the codebase.

## Quick Start

```bash
# Install Claude Code (if you haven't)
npm install -g @anthropic-ai/claude-code

# Navigate to project
cd multi-agent-design-patterns

# Start Claude Code
claude-code
```

## Project Overview

This is a collection of **multi-agent design patterns** implementations, demonstrating various coordination and architecture strategies for LLM-powered agents.

**Key Areas:**
- Agent pattern implementations (lessons)
- Multi-agent framework studies (LangChain, AgentScope, DeepAgent, etc.)
- Memory solutions (Mem0, Zep, Evermemos)
- LLM training resources (SFT, DPO)
- RAG implementations

## Useful Claude Prompts

### Learning the Codebase
```
"Explain how the agent coordination works in lesson1_ordinary_agent"
"What's the difference between AgentScope and DeepAgent frameworks?"
"Show me the LangChain integration patterns"
```

### Making Changes
```
"Add a new agent pattern following the existing lesson structure"
"Create a new agent that uses LangGraph for state management"
"Add error handling to the base_context.py file"
```

### Running & Debugging
```
"Run the lesson1 example and debug any errors"
"Check if the code follows ruff formatting rules"
"Why is the agent not responding to my input?"
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python >=3.13 |
| Package Manager | `uv` |
| Agent Framework | LangChain, LangGraph |
| Database | PostgreSQL + pgvector |
| Linting | ruff (line-length: 100) |

## Common Commands

```bash
# Install dependencies
uv sync

# Run a script
uv run python <path_to_script.py>

# Run tests
uv run pytest -q

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
multi-agent-design-patterns/
├── multi-agent-design-patterns/    # Pattern implementations (by lesson)
├── multi-agent-framework/          # Framework studies
├── multi-agent-memory/             # Memory solutions
├── llm-lab/                        # LLM training resources
├── llm-rag/                        # RAG implementations
├── llm-memory/                     # Memory research
└── vibecoding-workshop/            # Workshop materials
```

## Coding Conventions

- **Language**: Chinese (中文) for documentation/comments
- **Changes**: Minimal, targeted edits
- **Style**: Follow existing patterns in the codebase
- **Linting**: ruff with line-length 100
- **Secrets**: Use `.env` (copy from `.env.example`)

## Tips for Best Results

1. **Read first, edit second** - Let Claude understand the existing code before making changes
2. **Use specific prompts** - Be clear about what you want to change
3. **Run validation** - Ask Claude to run tests or linting after changes
4. **Check the context** - Claude has access to `.claude.md` for project context

## Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Then run:
uv sync
```

## Need Help?

- Check `AGENTS.md` for detailed agent development guidelines
- See `README.md` for project overview
- Open an issue on GitHub
