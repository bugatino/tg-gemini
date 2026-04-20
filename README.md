# tg-gemini

<div align="center">

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/atticuszeller/tg-gemini)
[![Test](https://github.com/atticuszeller/tg-gemini/actions/workflows/main.yml/badge.svg)](https://github.com/atticuszeller/tg-gemini/actions/workflows/main.yml)

**Telegram ↔ Gemini CLI 中间件**

在 VPS 上持续运行，将 Telegram 消息转发给 Gemini CLI，实时流式返回响应。

</div>

---

```
手机 Telegram  ──→  tg-gemini (VPS)  ──→  gemini CLI (本地)
     ↑                                          │
     └──────────── 流式响应 ←───────────────────┘
```

## 前置条件

- **Python 3.12+** 和 **[uv](https://docs.astral.sh/uv/)**
- **Gemini CLI** 已安装并完成登录：
  ```bash
  npm install -g @google/gemini-cli
  gemini   # 首次运行完成认证
  ```
- **Telegram Bot Token**：通过 [@BotFather](https://t.me/botfather) 创建

## 安装

```bash
# pip
pip install tg-gemini

# uv（推荐，全局工具安装）
uv tool install tg-gemini

# 源码开发模式
git clone https://github.com/atticuszeller/tg-gemini
cd tg-gemini && uv sync --all-groups
```

## 快速开始

```bash
# 1. 创建配置目录
mkdir -p ~/.tg-gemini

# 2. 复制并编辑配置（详细注释见 config.example.toml）
cp config.example.toml ~/.tg-gemini/config.toml

# 编辑：至少填写 telegram.token
# [telegram]
# token = "123456:ABC-DEF..."

# 3. 启动
tg-gemini start

# 或指定配置文件
tg-gemini start --config /path/to/config.toml
```

**Ctrl+C** 优雅停止。

## 配置

配置文件为 TOML 格式，路径解析顺序：

1. `--config` 参数
2. 当前目录 `config.toml`
3. `~/.tg-gemini/config.toml`（默认）

最小配置：

```toml
[telegram]
token = "YOUR_BOT_TOKEN"
```

完整选项见 [`config.example.toml`](config.example.toml) 和 [docs/configuration.md](docs/configuration.md)。

### 常用配置片段

```toml
# 限制只有自己能用（向 @userinfobot 查询 ID）
[telegram]
token = "..."
allow_from = "你的用户ID"

# 设置工作目录和模式
[gemini]
work_dir = "/path/to/your/project"
mode = "yolo"   # 全自动，推荐服务端使用
```

## Bot 命令

| 命令 | 功能 |
|------|------|
| `/new` | 开启新 Gemini 会话 |
| `/list` | 查看所有会话 |
| `/switch <目标>` | 切换会话 |
| `/name <名称>` | 重命名会话 |
| `/delete` | 删除会话 |
| `/history` | 查看对话历史 |
| `/status` | 当前状态信息 |
| `/model [名称]` | 查看/切换模型 |
| `/mode [模式]` | 查看/切换工具模式 |
| `/lang [en\|zh]` | 切换界面语言 |
| `/quiet` | 切换静音（隐藏工具通知）|
| `/stop` | 终止当前 Agent |
| `/help` | 帮助 |

完整说明见 [docs/commands.md](docs/commands.md)。

## 群聊支持

```toml
[telegram]
token = "..."
group_reply_all = false       # 仅响应 @bot 或回复 bot（默认）
share_session_in_channel = false  # 每人独立会话（默认）
```

默认行为：只有 @机器人、回复机器人的消息、以及 `/command` 会被处理。

## 作为服务运行

创建 `/etc/systemd/system/tg-gemini.service`：

```ini
[Unit]
Description=tg-gemini Telegram Bot
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/home/ubuntu/.local/bin/tg-gemini start
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tg-gemini
sudo journalctl -u tg-gemini -f
```

## 架构

tg-gemini 由 7 个核心模块组成：

```
cli.py → engine.py → gemini.py       子进程 + JSONL 解析
              ↕
  telegram_platform.py               长轮询 + 消息收发
  streaming.py                       流式预览节流
  session.py                         多会话 + 历史持久化
  card.py                            卡片 UI（InlineKeyboard）
```

详细架构图和数据流见 [docs/architecture.md](docs/architecture.md)。

## 常见问题

**Q: Gemini CLI 没有响应**

```bash
gemini -p "hello" --output-format stream-json  # 测试 CLI 是否正常
```

**Q: 收不到 Telegram 消息**

检查 `allow_from` 配置，查看日志：`tg-gemini start` 或 `journalctl -u tg-gemini -f`。

**Q: 流式预览触发限流**

增大 `interval_ms`：
```toml
[stream_preview]
interval_ms = 3000
```

**Q: 如何让 Gemini 记住上下文**

不要发 `/new`。服务重启后自动恢复到最近一次会话（通过 `--resume` 机制）。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/configuration.md](docs/configuration.md) | 完整配置项参考表 |
| [docs/commands.md](docs/commands.md) | 所有 Bot 命令详细说明 |
| [docs/architecture.md](docs/architecture.md) | 组件架构、数据流、设计决策 |
| [docs/development.md](docs/development.md) | 开发环境、测试、发布流程 |
| [docs/internals.md](docs/internals.md) | stream-json 协议、Markdown 转换、流式预览实现 |
| [config.example.toml](config.example.toml) | 带注释的完整配置示例 |

## 开发

```bash
git clone https://github.com/atticuszeller/tg-gemini
cd tg-gemini && uv sync --all-groups

bash dev.sh check    # format → lint → test → pre-commit
bash dev.sh test     # 运行测试（662 个，98.95% 覆盖率）
bash dev.sh lint     # ty + ruff
```

详见 [docs/development.md](docs/development.md)。

## License

MIT
