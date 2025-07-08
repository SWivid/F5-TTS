## F5-TTS 项目指南

### 1. 项目概述

本项目是 F5-TTS，一个基于 Flow Matching 的文本转语音（TTS）深度学习模型。旨在生成流畅、忠实的语音。主要源代码位于 `src/f5_tts` 目录。

### 2. 技术栈与规范

- **语言**: Python 3.10
- **包管理**: 使用 `pip` 和 `pyproject.toml`。通过 `pip install -e .` 在本地进行可编辑模式的安装。
- **代码风格与检查**:
    - **工具**: 项目使用 `ruff` 进行代码格式化、导入排序和 linting。
    - **配置**: 规则定义在 `ruff.toml` 文件中。行长度限制为 120 个字符。
    - **提交流程**: 项目使用 `pre-commit` 在代码提交前自动运行 `ruff` 检查。任何代码修改都必须通过这些检查。

### 3. 关键命令

- **安装依赖**:
  ```bash
  pip install -e .
  ```
- **安装 pre-commit 钩子**:
  ```bash
  pre-commit install
  ```
- **手动运行代码检查**:
  ```bash
  # 运行所有 pre-commit 钩子
  pre-commit run --all-files

  # 或者单独运行 ruff
  ruff format .
  ruff check --fix .
  ```
- **运行 Gradio 应用 (推理)**:
  ```bash
  f5-tts_infer-gradio
  ```
- **运行命令行推理**:
  ```bash
  f5-tts_infer-cli -c <path_to_config.toml>
  ```
- **运行 Gradio 应用 (微调)**:
  ```bash
  f5-tts_finetune-gradio
  ```

### 4. 代码修改指南

- **遵循现有风格**: 所有代码修改都应严格遵守 `ruff` 所定义的现有代码风格和格式。
- **使用 `pre-commit`**: 在提交（commit）代码之前，请务必运行 `pre-commit run --all-files` 以确保代码质量。
- **命令行工具**: 项目通过 `pyproject.toml` 的 `[project.scripts]` 定义了多个命令行入口点，如 `f5-tts_infer-cli`。在添加新脚本时，应遵循此模式。
- **配置文件**: 项目广泛使用 `.toml` 和 `.yaml` 文件进行配置（如推理和模型配置）。修改时请注意其结构。

### 5. 提交信息

- 提交信息应清晰、简洁，并能准确描述所做的更改。可以参考 `git log` 中现有的提交信息风格。
