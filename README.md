# EchQ

一个基于 Python 的智能 QQ 聊天机器人框架，通过 NapCat 框架与 QQ 进行交互，支持大语言模型 (LLM) 驱动的对话功能

## ✨ 特性

- 🤖 **智能对话**：基于大语言模型的自然语言交互
- 💬 **多场景支持**：支持私聊和群聊消息处理
- 🧠 **上下文记忆管理**：智能管理对话上下文，支持 token 限制和自动清理
- 💰 **缓存优化**：可选的缓存管理功能，降低 API 调用成本
- ⚡ **指令系统**：内置指令支持，方便查看状态和管理机器人
- 🔄 **流式响应**：支持流式输出，实时返回 AI 回复

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
    LLM_API_URL = 'https://api.openai.com/v1'  # API 地址
    LLM_API_KEY = 'your_api_key_here'          # API 密钥
    LLM_MODEL = 'gpt-4o'                       # 模型名称
    LLM_TEMPERATURE = 1.3                      # 温度参数
    
    # NapCat 配置
    NAPCAT_HTTP_URL = 'http://localhost:3000'  # HTTP 接口地址
    NAPCAT_WS_URL = 'ws://localhost:3001'      # WebSocket 地址
    
    # 其他配置项...
```

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

### Agent 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `AGENT_CAN_SEE_DATETIME` | 是否可以看到当前日期时间 | `False` |

### 记忆管理配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `AMEM_TOKEN_LIMIT` | 上下文记忆的最大 token 限制 | `64000` |
| `AMEM_EXPECTED_TOKEN_USAGE` | 期望的 token 使用量 | `16000` |
| `AMEM_ENABLE_CACHE_MANAGEMENT` | 是否启用缓存管理 | `False` |
| `AMEM_CACHE_EXPIRY_SECONDS` | 缓存过期时间（秒） | `300` |
| `AMEM_CACHE_PRICE_RATIO` | 缓存价格比率 | `0.5` |

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

## 🏗️ 项目结构
```
EchQ/
├── app.py                      # 主程序入口
├── config/
│   ├── config.py.example      # 配置模板
│   └── prompt.txt.example     # 提示词模板
├── agent/
│   ├── agent.py                # Agent 核心逻辑
│   └── agent_memory.py         # 记忆管理模块
├── napcat/
│   ├── napcat.py               # NapCat 客户端
│   └── message_formatter.py    # 消息格式化工具
└── requirements.txt            # 项目依赖
```

## 🔧 开发指南

### 自定义 AI 人格

编辑 `config/prompt.txt` 文件，可以定制 AI 提示词

### 扩展功能

你可以通过修改以下模块来扩展功能：
- `agent/agent.py` - 添加新的 Agent 能力
- `app.py` - 添加新的指令或消息处理逻辑
- `napcat/message_formatter.py` - 自定义消息格式化规则

## ⚠️ 注意事项

1. 请确保 API 密钥的安全，不要将 `config.py` 提交到公开仓库
2. 根据使用的 LLM 服务调整 token 限制配置
3. 本项目适用于私聊与小型群聊，大型群聊场景下注意控制回复频率，避免刷屏
4. 定期检查 token 使用量，控制 API 成本

## 🚧 开发进度

这是一个个人学习项目，目前处于早期开发阶段

### ✅ 已实现功能

- [x] 基础对话功能（私聊/群聊）
- [x] 上下文记忆管理
- [x] Token 使用量监控
- [x] 流式响应输出
- [x] 基础指令系统（/help, /context, /token）
- [x] 可缓存的上下文管理
- [x] WebSocket 消息监听
- [x] 自定义 AI 人格配置

### 📋 开发计划

#### 近期目标

- [ ] **功能增强**
  - [ ] 拓展指令集
  - [ ] QQ 指令权限分级 (管理员/好友/访客)
  - [ ] 好友备注功能
  - [x] 处理并发消息

- [ ] **记忆系统**
  - [ ] 启动时读取历史记忆
  - [ ] 长期记忆

#### 中期目标

- [ ] **多模态支持**
  - [ ] 读取图片
  - [ ] 生成图片
  - [ ] 语音消息支持

- [ ] **富媒体消息**
  - [ ] 发送 QQ emoji 表情
  - [ ] 发送音效
  - [ ] 发送表情包

#### 长期目标

- [ ] **高级功能**
  - [ ] 工具调用能力

- [ ] **高级记忆系统**
  - [ ] 向量数据库集成
  - [ ] 图数据库集成

## 📝 许可证

本项目采用 [MIT License](LICENSE) 开源

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

邮箱: CountlessBugs@outlook.com
