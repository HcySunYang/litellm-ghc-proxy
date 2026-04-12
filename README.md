# LiteLLM GitHub Copilot Proxy

通过 [LiteLLM](https://github.com/BerriAI/litellm) 代理 GitHub Copilot 的 AI 模型，提供 OpenAI 兼容的 API 接口。一个账号即可访问 Claude、GPT、Gemini 等多家模型。

## 支持的模型

| 厂商 | 模型 |
|------|------|
| **Anthropic** | Claude Opus 4.6 / 4.5 / 4.1, Claude Sonnet 4.5 / 4, Claude Haiku 4.5 |
| **OpenAI** | GPT-5.2 / 5.1 / 5 / 5-mini, GPT-5.1-Codex / Codex-Max, GPT-5.2-Codex, GPT-4.1 / 4o / 4o-mini / 4, GPT-3.5-turbo |
| **Google** | Gemini 2.5 Pro |

## 前置要求

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- Python 3
- 拥有 GitHub Copilot 订阅的 GitHub 账号
- (可选) `jq` — 用于列出可用模型

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/codetrek/litellm-ghc-proxy.git
cd litellm-ghc-proxy
```

### 2. 生成环境变量

```bash
python3 generate_env.py
```

会自动生成 `.env` 文件，包含 API Master Key 和管理后台密码。请**记录终端输出的密钥信息**：

```
Master Key: litellm-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Admin Password: xxxxxxxxxxxxxxxx
```

> 如果 `.env` 已存在则会跳过，不会覆盖。如需重新生成，请先删除 `.env`。

### 3. (可选) 配置代理

如果你的网络环境需要通过代理访问 GitHub，编辑 `.env` 文件添加：

```bash
HTTPS_PROXY=http://your-proxy:port
HTTP_PROXY=http://your-proxy:port
```

> macOS/Windows 上 Docker Desktop 可使用 `http://host.docker.internal:port` 访问宿主机代理。

### 4. 启动服务

```bash
docker compose up -d
```

首次启动会拉取 Docker 镜像（litellm + PostgreSQL 16），并自动进入 GitHub 设备认证流程。

### 5. 完成 GitHub 认证

查看日志获取认证码：

```bash
docker logs -f litellm-proxy
```

日志中会出现类似提示：

```
Please open https://github.com/login/device and enter code: XXXX-XXXX
```

1. 浏览器打开 https://github.com/login/device
2. 输入日志中显示的认证码
3. 授权后，token 会自动保存到 `litellm-data/github_copilot/access-token`

认证成功后服务即可正常使用。

### 6. 验证服务

```bash
./test-proxy.sh
```

或手动测试：

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <你的 LITELLM_MASTER_KEY>" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 使用方式

### API 接口

服务启动后，即可通过 OpenAI 兼容 API 调用：

- **Base URL:** `http://localhost:4000`
- **Chat Completions:** `POST /chat/completions`
- **认证方式:** `Authorization: Bearer <LITELLM_MASTER_KEY>`

可以在任何支持 OpenAI API 格式的客户端/工具中使用，只需将 base URL 指向 `http://localhost:4000`。

### Web 管理界面

访问 http://localhost:4000/ui ，使用以下凭据登录：

- **用户名:** `ImNotAdmin`
- **密码:** `generate_env.py` 生成时输出的 Admin Password

### API 文档

访问 http://localhost:4000/docs 查看 Swagger API 文档。

## 常用命令

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 查看日志
docker logs -f litellm-proxy

# 重启代理（修改配置后）
docker compose restart ghc-proxy

# 列出 GitHub Copilot 所有可用模型
./list-copilot-models.sh

# 仅列出已启用的模型
./list-copilot-models.sh --enabled-only

# 测试代理连通性
./test-proxy.sh
```

## 自定义模型配置

编辑 `conf/copilot-config.yaml`，添加或移除模型条目：

```yaml
model_list:
  - model_name: claude-sonnet-4
    litellm_params:
      model: github_copilot/claude-sonnet-4
      extra_headers:
        copilot-integration-id: "copilot-chat"
        x-github-api-version: "2025-05-01"
```

> **重要:** `extra_headers` 中的 header 名必须使用**小写**（如 `copilot-integration-id`），以正确覆盖 LiteLLM 内部默认值。大小写不一致会导致发送重复 header，引发 API 错误。

运行 `./list-copilot-models.sh` 可查看所有可用模型及其参数，将需要的模型复制到配置文件中。修改后重启服务生效：

```bash
docker compose restart ghc-proxy
```

## 项目结构

```
.
├── docker-compose.yml          # Docker 编排配置
├── generate_env.py             # 环境变量生成脚本
├── test-proxy.sh               # API 连通性测试脚本
├── list-copilot-models.sh      # 列出可用模型的工具脚本
├── .env                        # 环境变量（自动生成，不入库）
├── conf/
│   ├── copilot-config.yaml     # LiteLLM 模型配置
│   └── postgres.conf           # PostgreSQL 配置
└── litellm-data/               # 运行时数据（不入库）
    ├── github_copilot/         # GitHub Copilot 认证信息
    └── postgres/               # PostgreSQL 数据
```

## 使用 OpenAI Codex CLI

除了 Claude Code，本项目也支持 [OpenAI Codex CLI](https://github.com/openai/codex) 通过同一个代理访问模型。

### 1. 安装 Codex CLI

```bash
npm install -g @openai/codex
# 或
brew install --cask codex
```

### 2. 配置环境变量

将 LiteLLM Master Key 设为环境变量（添加到 `~/.bashrc` 或 `~/.zshrc`）：

```bash
export LITELLM_API_KEY="<你的 LITELLM_MASTER_KEY>"
```

### 3. 配置 Codex

创建或编辑 `~/.codex/config.toml`：

```toml
model = "gpt-5.2-codex"
model_provider = "litellm-proxy"

[model_providers.litellm-proxy]
name = "LiteLLM GitHub Copilot Proxy"
base_url = "http://<服务器IP>:4000"
env_key = "LITELLM_API_KEY"
wire_api = "responses"
```

> `model` 可替换为 `copilot-config.yaml` 中的任何模型名，如 `gpt-5.4`、`gpt-5.3-codex`、`claude-sonnet-4` 等。

> 如果代理运行在云服务器上（如 Azure VM），需要在安全组/防火墙中放行 **TCP 4000** 端口。

### 4. 启动 Codex

```bash
codex
```

> Codex CLI 直接连接 LiteLLM（端口 4000），不经过中间件。Claude Code 仍通过中间件（端口 80）使用，互不影响。

## 注意事项

- 服务默认监听 `0.0.0.0`。端口 80 为中间件（Claude Code 使用），端口 4000 为 LiteLLM 直连（Codex CLI 使用）。云服务器部署时，请在安全组中按需放行端口，并确保使用强密钥
- `.env` 和 `litellm-data/` 包含敏感信息，已在 `.gitignore` 中排除
- GitHub Copilot token 会过期，过期后重启服务会自动触发重新认证
