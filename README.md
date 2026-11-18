# LiteLLM GitHub Copilot Proxy

通过 [LiteLLM](https://github.com/BerriAI/litellm) 代理 GitHub Copilot 的 AI 模型，提供 OpenAI 兼容的 API 接口。一个账号即可访问 Claude、GPT、Gemini 等多家模型。

## 支持的模型

| 厂商 | 模型 |
|------|------|
| **Anthropic** | Claude Opus 4.7 / 4.6 / 4.5 / 4.1, Claude Sonnet 4.6 / 4.5 / 4, Claude Haiku 4.5 |
| **OpenAI** | GPT-6.2 Sol, GPT-6.1 Sol, GPT-6 Astra, GPT-5.6-Luna / Sol / Terra, GPT-5.5, GPT-5.4 / 5.4-mini, GPT-5.3-Codex, GPT-5.2 / 5.2-Codex, GPT-5.1 / 5.1-Codex / Codex-Max, GPT-5 / 5-mini, GPT-4.1 / 4o / 4o-mini / 4, GPT-3.5-turbo |
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
- **Responses API:** `POST /responses`
- **认证方式:** `Authorization: Bearer <LITELLM_MASTER_KEY>`

> **注意：** GPT-5.5 及更新的模型（包括 GPT-5.6-Luna/Sol/Terra、GPT-6 Astra、GPT-6.1 Sol、GPT-6.2 Sol）仅支持 Responses API（`/responses` 端点），不支持 `/chat/completions`。Codex CLI 等已支持 Responses API 的客户端可直接使用。GPT-6 Astra 的模型 ID 为 `gpt-6-astra`，GPT-6.1 Sol / GPT-6.2 Sol 分别为 `gpt-6.1-sol` / `gpt-6.2-sol`。

例如，以最高推理级别调用 GPT-6.2 Sol：

```bash
curl http://localhost:4000/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <你的 LITELLM_MASTER_KEY>" \
  -d '{
    "model": "gpt-6.2-sol",
    "input": "Hello!",
    "reasoning": {"effort": "max"}
  }'
```

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

> **长会话限制：** Copilot 的 Responses API 请求体最多为 32 MiB，这与模型的
> token 上下文窗口是两个独立限制。大量工具输出、图片或加密推理记录可能先触发
> HTTP 413，上游错误文本为 `failed to parse request`。中间件将这类错误转换为
> Codex 可识别的 `context_length_exceeded`，让客户端按上下文超限处理；代理不会
> 自行截断会话。已超限的旧会话可先执行 `/compact` 再重试；若单条输入或工具输出
> 本身过大，需要减少该输入，或创建新会话。

在另一台机器配置 GPT-6 Astra 客户端，请参阅
[Codex CLI + GPT-6 Astra 配置文档](docs/codex-gpt-6-astra.md)。

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
model = "gpt-5.6-luna"
model_provider = "litellm-proxy"

[model_providers.litellm-proxy]
name = "LiteLLM GitHub Copilot Proxy"
base_url = "http://<服务器IP>:4000"
wire_api = "responses"

[model_providers.litellm-proxy.auth]
command = "printenv"
args = ["LITELLM_API_KEY"]
```

> `model` 可替换为 `copilot-config.yaml` 中的任何模型名，如 `gpt-6.2-sol`、`gpt-6.1-sol`、`gpt-6-astra`、`gpt-5.4`、`gpt-5.3-codex`、`claude-sonnet-4` 等。

> **关于 context window：** 代理会在 Codex 请求
> `/models?client_version=...` 时返回 Codex 专用模型目录。GPT-5.6-Luna /
> Sol / Terra 会报告 `1050000` 原始 context，并将 Codex 有效窗口设为
> `913500`（87%）、自动压缩阈值设为 `900000`，从而保持在后端
> `922000` 最大输入和 `128000` 最大输出限制内。无需在本地设置 `model_context_window` 或
> `model_catalog_json`。这里使用 command auth 是因为 Codex 只会为这类自定义
> provider 主动刷新 `/models`；`printenv` 仍然只是读取前面设置的
> `LITELLM_API_KEY`。如果此前配置过 `model_catalog_json`，请删除该项；升级代理后
> 重启 Codex 并创建新会话，使代理返回的模型目录重新加载。

> **GPT-6 Astra：** 将上述 `model` 设为 `gpt-6-astra`。该模型的原始 context 为
> `1000000`，最大输入为 `872000`，最大输出为 `128000`；代理将 Codex 有效窗口设为
> `870000`（87%）、自动压缩阈值设为 `850000`。支持 `low`、`medium`、`high`、
> `xhigh`、`max` 推理级别，其中 `max` 仅向 Codex 0.144.0 及更新客户端提供。

> **GPT-6.1 Sol：** 将上述 `model` 设为 `gpt-6.1-sol`。该模型的原始 context 为
> `1050000`，最大输入为 `922000`，最大输出为 `128000`；代理将 Codex 有效窗口设为
> `913500`（87%）、自动压缩阈值设为 `900000`。默认推理级别为 `medium`，
> Codex 目录提供 `low`、`medium`、`high`、`xhigh`，并向 0.144.0 及更新客户端提供
> `max`；不提供 `ultra`。直接调用 Responses API 时也可使用 `none`。

> **GPT-6.2 Sol：** 将上述 `model` 设为 `gpt-6.2-sol`。上游报告的原始 context 和
> 最大输入均为 `1050000`，最大输出为 `128000`。为输出预留空间，代理将 Codex
> 有效窗口设为 `913500`（87%）、自动压缩阈值设为 `900000`。默认推理级别为
> `medium`，支持 `low`、`medium`、`high`、`xhigh`、`max` 五档，不提供 `ultra`。
> 沿用客户端兼容规则：`max` 仅向 Codex 0.144.0 及更新客户端展示，直接调用
> Responses API 不受此版本限制；Codex 中可设置 `model_reasoning_effort = "max"`。

已有部署更新后，需要重新加载模型配置并重建模型目录所在的 middleware：

```bash
docker compose restart ghc-proxy
docker compose up -d --build --no-deps middleware
```

> 如果代理运行在云服务器上（如 Azure VM），需要在安全组/防火墙中放行 **TCP 4000** 端口。

### 4. 启动 Codex

```bash
codex
```

> Codex CLI 使用端口 4000，Claude Code 使用端口 80；两个端口都由中间件接入，
> 再转发到内部 LiteLLM。中间件只会为 Codex 的模型目录请求补充元数据，其余
> OpenAI 请求保持透明转发。

> **仅 Codex CLI / IDE 扩展支持本代理。** ChatGPT 桌面应用内置的 Codex 使用 ChatGPT 账号登录与 OpenAI 官方后端，**不读取 `~/.codex/config.toml` 的 `model_provider`/`base_url`**，因此无法指向本代理。若需在编辑器中使用，请改用 VS Code / Cursor 的 Codex 扩展并沿用上面的 `config.toml` 配置。

## 注意事项

- 服务默认监听 `0.0.0.0`。端口 80 和 4000 均由中间件接入，LiteLLM 仅在 Docker 内部网络监听。云服务器部署时，请在安全组中按需放行端口，并确保使用强密钥
- `.env` 和 `litellm-data/` 包含敏感信息，已在 `.gitignore` 中排除
- GitHub Copilot token 会过期，过期后重启服务会自动触发重新认证
