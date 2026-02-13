---
description: Git 提交信息规范，遵循 Conventional Commits 约定
---

# Git 提交规范工作流

## 提交格式

```
<type>(<scope>): <description>
```

## 步骤

### 1. 确定变更类型（type）

根据变更内容选择合适的 type：

- `feat` — 新增功能或代码特性
- `fix` — 修复 Bug
- `docs` — 仅文档变更（README、注释、说明文件）
- `style` — 代码格式调整（不影响逻辑）
- `refactor` — 代码重构（不新增功能也不修复 Bug）
- `test` — 测试相关
- `chore` — 构建、依赖、配置等杂项
- `perf` — 性能优化
- `ci` — CI/CD 配置变更
- `build` — 构建系统或外部依赖变更
- `revert` — 回退之前的提交

### 2. 确定作用域（scope，可选）

使用模块名作为 scope，例如：`evermemos`、`mem0`、`zep`、`multi-agent-memory`

### 3. 编写描述

- 使用中文
- 动词开头：创建、添加、更新、修复、移除、重构、优化
- 简洁明了，不超过 50 字符
- 不以句号结尾

### 4. 执行提交

// turbo
```bash
git add .
```

```bash
git commit -m "<type>(<scope>): <description>"
```

## 常用操作速查

### 文档（README）操作

| 操作 | 提交命令 |
|------|----------|
| 创建 | `git commit -m "docs(<module>): 创建 README 文件"` |
| 更新 | `git commit -m "docs(<module>): 更新 README 中的 XXX 说明"` |
| 修正 | `git commit -m "docs(<module>): 修正 README 中的错误链接"` |
| 删除 | `git commit -m "docs: 移除 <module> 中过期的 README"` |

### 功能操作

| 操作 | 提交命令 |
|------|----------|
| 新增功能 | `git commit -m "feat(<module>): 添加 XXX 功能"` |
| 修复 Bug | `git commit -m "fix(<module>): 修复 XXX 的问题"` |
| 重构代码 | `git commit -m "refactor(<module>): 重构 XXX 模块"` |
| 删除代码 | `git commit -m "refactor(<module>): 移除 XXX 模块"` |

### 配置/依赖操作

| 操作 | 提交命令 |
|------|----------|
| 更新依赖 | `git commit -m "chore: 更新 pyproject.toml 依赖版本"` |
| 修改配置 | `git commit -m "chore: 更新 .gitignore 规则"` |
| 环境变量 | `git commit -m "chore: 添加 .env.example 模板"` |
