# EchQ

一个基于 Python 的 QQ 聊天机器人应用，目标是构建与真实人类行为相似的 Agent，采用 LangGraph 图状态机架构，通过 NapCat 框架与 QQ 进行交互，支持大语言模型 (LLM) 驱动的对话功能

## ✨ 特性

- 🤖 **智能对话**：基于大语言模型的自然语言交互
- 🧠 **上下文记忆管理**：智能管理对话上下文，支持 token 限制和自动清理
- ⚡ **指令系统**：内置指令支持，方便查看状态和管理机器人
- 🔄 **流式响应**：支持流式输出，实时返回 AI 回复
- 🛠️ **模块化设计**：基于LangGraph的节点化架构，便于功能扩展

## 📋 系统要求

- Python 3.10+
- NapCat (QQ 机器人框架实现)
- 支持 OpenAI API 格式的 LLM 服务

## 🚀 快速开始

### 1. 安装和配置 NapCat

NapCat 是一个现代化的 QQ Bot 协议框架，需要先安装配置才能使用本项目。

#### 安装 NapCat

请访问 [NapCat 官方文档](https://napneko.github.io/) 查看详细的安装教程。

NapCat 提供多种安装方式：

- **Shell 方式**：适合服务器部署，内存占用低（50-100MB），推荐使用
  - 📖 [Shell 安装教程](https://napneko.github.io/guide/boot/Shell)
  
- **Framework 方式**：作为 LiteLoader 插件运行，方便人机交互
  - 📖 [Framework 安装教程](https://napneko.github.io/guide/boot/Framework)

#### 配置 NapCat

1. 启动 NapCat 后，访问 WebUI 管理界面
2. 配置 HTTP 和 WebSocket 服务：
   - HTTP 服务地址（默认：`http://localhost:3000`）
   - WebSocket 服务地址（默认：`ws://localhost:3001`）
3. 记录这些地址，后续需要填入 EchQ 配置文件

更多配置信息请参考 [NapCat 配置文档](https://napneko.github.io/config/basic)

### 2. 克隆项目
```bash
git clone https://github.com/CountlessBugs/EchQ
cd EchQ
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置项目

#### 配置文件设置

将 `config.py.example` 复制为 `config.py`

编辑 `config/config.py`，填入你的配置：
```python
class Config:
    # LLM 配置
    LLM_MODEL = 'gpt-4o'                       # 模型名称
    LLM_TEMPERATURE = 1.3                      # 温度参数
    
    # NapCat 配置
    NAPCAT_HTTP_URL = 'http://localhost:3000'  # HTTP 接口地址
    NAPCAT_WS_URL = 'ws://localhost:3001'      # WebSocket 地址
    
    # 其他配置项...
```

将 `.env.example` 复制为 `.env`

编辑 `.env`，填入 API 配置
```env
# OpenAI 兼容接口
OPENAI_API_KEY=your_openai_key
# 可选：自定义 API 端点 (如使用第三方服务)
# OPENAI_API_BASE=https://api.openai.com/v1
```

适配的模型提供商详见 https://python.langchain.com/docs/integrations/chat/
注：目前仅支持 OpenAI ，未来提供其他供应商支持

#### 提示词设置

将 `prompt.txt.example` 复制为 `prompt.txt`

根据需要编辑 `config/prompt.txt`，定制你的 AI 角色人格和行为规则

### 5. 启动 NapCat

确保 NapCat 服务已启动并正确配置 HTTP 和 WebSocket 端口

### 6. 运行程序
```bash
python app.py
```

## 📖 使用说明

### QQ 指令

在 QQ 中可以使用以下指令（需启用 `ENABLE_COMMANDS`）：

- `/help` - 显示帮助信息
- `/context` - 查看当前上下文记忆
- `/token` - 查看当前上下文 token 使用量

## ⚙️ 配置说明

### LLM 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_API_URL` | LLM API 地址 | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `LLM_TEMPERATURE` | 温度参数（控制随机性） | `1.3` |

<!-- ### Agent 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------| -->

### 记忆管理配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `AMEM_TOKEN_LIMIT` | 上下文记忆的最大 token 限制 | `64000` |

### NapCat 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `NAPCAT_HTTP_URL` | NapCat HTTP 接口地址 | `http://localhost:3000` |
| `NAPCAT_WS_URL` | NapCat WebSocket 地址 | `ws://localhost:3001` |

### 其他配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ENABLE_COMMANDS` | 是否启用 QQ 指令功能 | `True` |
| `FILTER_WS_HEARTBEAT` | 是否过滤 WebSocket 心跳日志 | `True` |
| `PRINT_WS_MESSAGES` | 是否打印 Websocket 消息 | `False` |

## 🏗️ 项目结构
```
EchQ/
├── app.py                      # 主程序入口
├── config/
│   ├── config.py.example       # 配置模板
│   └── prompt.txt.example      # 提示词模板
├── agent/
│   ├── nodes/                  # Agent 功能节点
│   │   ├── nodes.py.example    # 节点模板
│   │   ├── basic_nodes.py      # 基础节点
│   │   └── llm_nodes.py        # LLM 节点
│   ├── workflows/              # Agent 图工作流
│   │   ├── wf.py.example       # 工作流模板
│   │   └── default_wf.py       # 默认工作流
│   ├── agent.py                # Agent 核心逻辑
│   └── agent_state.py          # Agent 状态定义
├── napcat/
│   ├── napcat.py               # NapCat 客户端
│   └── message_formatter.py    # 消息格式化工具
├── .env.example                # 环境变量模板
└── requirements.txt            # 项目依赖
```

## 🔧 开发指南

### 自定义 AI 人格

编辑 `config/prompt.txt` 文件，可以定制 AI 提示词

### 扩展功能

你可以通过修改以下模块来扩展功能：
- `agent/agent.py` - 添加新的 Agent 状态变量
- `agent/nodes/` - 创建自定义节点，添加新功能
- `agent/workflows/` - 定义新的工作流逻辑
- `app.py` - 添加新的指令或消息处理逻辑，或加载其他工作流

扩展 Agent 功能的具体步骤如下：
1. 在 `agent/agent_state.py` 中添加需要的状态变量（如果需要）
2. 在 `agent/tools/` 目录下创建新的工具
3. 在 `agent/nodes/` 目录下创建新的节点
4. 在 `agent/workflows/` 目录下创建新的工作流
5. 在 `app.py` 中加载新的工作流，将 CompiledGraph 作为 Agent 初始化参数

> **📝 NOTE**  
> 由于 LangGraph 中子图无法直接移除父图的消息，故需要层层向上传递一个待移除的消息 ID 列表 `message_ids_to_remove`。子图节点的后面需要添加 `basic_nodes` 中的 `cleanup_node` 节点来实际移除消息。Agent 根图中的 `_exit_node` 已经实现了该功能，无需额外添加。

## ⚠️ 注意事项

1. 请确保 API 密钥的安全，不要将 `.env` 提交到公开仓库
2. 本项目适用于私聊与小型群聊，大型群聊场景下注意控制回复频率，避免刷屏；多个聊天同时进行可能效果不佳
3. 图片生成工具中 API 格式可能需要根据所用模型提供商进行调整

## 🚧 开发进度

这是一个个人学习项目，目前处于早期开发阶段

## 🛠️ 功能模块

### 📝 记忆系统
- [x] 上下文记忆管理
- [ ] 长期记忆系统
- [ ] 向量数据库集成
- [ ] 图数据库集成

### 💬 对话与交互
- [x] 基础对话功能 (私聊/群聊)
- [x] 流式响应输出
- [x] 处理并发消息
- [x] WebSocket 消息监听

### ⚙️ 指令与权限
- [x] 基础指令系统 (/help, /context, /token)
- [ ] 拓展指令集
- [ ] QQ 指令权限分级（管理员/好友/访客）

### 🧠 AI 能力
- [x] 自动上下文总结
- [x] 工具调用能力

### 🎨 多模态支持
- [x] 生成图片
- [ ] 读取图片
- [ ] 语音消息支持
- [ ] 发送 QQ emoji 表情
- [ ] 发送音效
- [ ] 发送表情包

### 👥 社交功能
- [ ] 好友备注功能

### 📊 系统监控
- [x] Token 使用量监控

## 📝 许可证

本项目采用 [MIT License](LICENSE) 开源

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

邮箱: CountlessBugs@outlook.com
