# AgentScope / AgentScope 简介

![Banner](assets/LLM.jpg)

A concise implementation of **AgentScope**, a framework for orchestrating multi-agent systems with advanced language models.  
Reference: *Agentic Design Patterns*

AgentScope 是一个多智能体系统编排框架，专注于通过高级语言模型实现复杂任务的分解与执行。  
参考书籍：《Agentic Design Patterns》

---

## 📚 Project Introduction / 项目介绍

This is a specialized implementation of AgentScope, designed to streamline the coordination and communication of multiple intelligent agents in complex workflows.  

这是一个 AgentScope 的专项实现，旨在简化复杂工作流中多个智能体的协调与通信。

## 📖 Key Features / 主要特性

- 🔧 **Multi-Agent Orchestration** / 多智能体编排  
  Facilitates task decomposition and execution across multiple agents.  
  支持多智能体之间的任务分解与执行。

- 💡 **Dynamic Workflow Management** / 动态工作流管理  
  Enables dynamic adjustments to workflows based on real-time inputs.  
  根据实时输入动态调整工作流。

- 💬 **Language Model Integration** / 语言模型集成  
  Provides seamless integration with large language models (LLMs).  
  提供与大语言模型（LLMs）的无缝集成。

- 🌐 **Shared LLM Instances** / 共享LLM实例  
  Ensures efficient resource utilization by sharing a single LLM instance among multiple agents.  
  通过在多个智能体之间共享单个LLM实例，确保资源高效利用。

## 🛠 Usage Instructions / 使用说明

1. **Installation** / 安装依赖  
   Run the following command to install dependencies:  
   执行以下命令安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. **Execution** / 运行代码  
   Execute the main script to start the AgentScope system:  
   执行主脚本以启动 AgentScope 系统：
   ```bash
   python main.py
   ```

3. **Testing** / 测试  
   Run unit tests to verify functionality:  
   执行单元测试以验证功能：
   ```bash
   pytest test_main.py
   ```

---

## 📂 Directory Structure / 目录结构

```
multi_agent_framework/
├── agentscope/               # AgentScope Implementation / AgentScope 实现
│   ├── README.md            # Documentation / 文档
│   ├── main.py              # Main logic / 主逻辑
│   └── utils.py             # Utility functions / 工具函数
```

---

## 🌟 Contribution Guidelines / 贡献指南

We welcome contributions from the community! Please follow these steps:  
我们欢迎社区贡献！请遵循以下步骤：

1. Fork the repository and create a new branch.  
   Fork 仓库并创建新分支。
2. Make your changes and ensure all tests pass.  
   进行修改并确保所有测试通过。
3. Submit a pull request with detailed descriptions.  
   提交包含详细描述的 Pull Request。
---