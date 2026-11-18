# 在另一台机器配置 Codex CLI + GPT-6 Astra

本文适用于 macOS、Linux，以及 Windows 的 WSL 环境。新机器只运行 Codex CLI，
连接已经部署好的 LiteLLM GitHub Copilot Proxy；不需要再安装 Docker、
部署代理或登录 GitHub Copilot。

GPT-6.1 Sol 也适用本文的连接和认证配置；将下文的 `gpt-6-astra` 替换为
`gpt-6.1-sol` 即可，包括配置文件、请求以及虚拟密钥权限。

当前服务端地址按项目配置为 `20.89.181.138`，代理监听端口为 `4000`。
如果服务器地址发生变化，请替换下文中的 IP。

## 1. 安装 Codex CLI

安装 Node.js LTS 后，在新机器执行：

```bash
npm install -g @openai/codex@latest
codex --version
```

macOS 也可以使用 `brew install --cask codex`，两种安装方式选一种即可。
建议使用最新版；本代理仅向 Codex `0.144.0` 及更新客户端提供 `max` 推理级别。

## 2. 连接代理服务器

推荐使用 SSH 隧道，避免通过公网 HTTP 明文传输密钥和对话内容。
在新机器单独打开一个终端，将 `<SSH用户名>` 替换为你的服务器登录用户名：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:14000:127.0.0.1:4000 \
  <SSH用户名>@20.89.181.138
```

保持该终端运行。此时新机器上的 `http://127.0.0.1:14000` 会连接到服务端代理，
无需对新机器开放服务端的公网 `4000` 端口，只需能通过 SSH 登录服务器。

本文后续配置默认使用这个隧道地址。如果采用其他连接方式，同时替换下文
`LITELLM_BASE_URL` 和 `config.toml` 中的 `base_url`：

| 连接方式 | Base URL | 注意事项 |
|----------|----------|----------|
| SSH 隧道，本文默认 | `http://127.0.0.1:14000` | 保持隧道终端运行 |
| 已配置的 HTTPS 反向代理 | `https://<你的代理域名>` | 需要已配置好域名、证书和转发 |
| HTTP 直连 | `http://20.89.181.138:4000` | 会明文传输密钥和对话，仅用于可信网络；防火墙仅放行所需客户端 IP |

注意：不使用隧道时，新机器上的 `localhost:4000` 不是这台远程代理服务器。

## 3. 设置代理访问密钥

访问密钥使用服务端项目 `.env` 中的 `LITELLM_MASTER_KEY`，
或者管理员签发的、允许访问 `gpt-6-astra` 的 LiteLLM 虚拟密钥。
它不是 GitHub Token、Copilot Token 或 OpenAI API Key。

在准备运行 Codex 的终端中设置：

```bash
export LITELLM_API_KEY="<替换为你的代理访问密钥>"
export LITELLM_BASE_URL="http://127.0.0.1:14000"
```

这里客户端变量叫 `LITELLM_API_KEY`，其值是上面的代理访问密钥。
`LITELLM_BASE_URL` 仅用于后文的 curl 命令，Codex 不会自动读取它。

以上设置仅对当前终端及其子进程生效。如需持久化，可放入你使用的
`~/.zshrc` 或 `~/.bashrc`，再打开一个新终端。不要把真实密钥提交到仓库、
粘贴到聊天记录或分享给他人。

## 4. 配置 Codex

创建配置目录：

```bash
mkdir -p ~/.codex
```

用编辑器创建或修改 `~/.codex/config.toml`，填入：

```toml
model = "gpt-6-astra"
model_provider = "litellm-proxy"
model_reasoning_effort = "medium"

[model_providers.litellm-proxy]
name = "LiteLLM GitHub Copilot Proxy"
base_url = "http://127.0.0.1:14000"
wire_api = "responses"
requires_openai_auth = false

[model_providers.litellm-proxy.auth]
command = "printenv"
args = ["LITELLM_API_KEY"]
```

如果文件已存在，只合并相关字段并保留其他设置，不要重复声明同名键或表。
`model`、`model_provider`、`model_reasoning_effort` 是顶层字段，应放在表头之前。
如果设置过 `CODEX_HOME`，请修改该目录内的 `config.toml`，而不是默认路径。

这里使用 command auth，让 Codex 能向自定义 provider 刷新模型目录；
`printenv` 只读取本机环境变量。不要为此 provider 同时保留旧的 `env_key`
认证配置，也不需要运行 `codex login` 来登录 ChatGPT。

无需手动设置 `model_context_window`、`model_auto_compact_token_limit`
或 `model_catalog_json`。如果以前配置过这些覆盖项，请移除与本次配置冲突的项，
让代理返回的模型元数据生效。

## 5. 确认连通性

在已经设置好环境变量的终端中，查看 Codex 专用模型目录：

```bash
curl --fail-with-body --silent --show-error \
  "$LITELLM_BASE_URL/models?client_version=0.144.0" \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

返回的 `models` 数组应包含 `"slug": "gpt-6-astra"`，
且该项的 `visibility` 为 `"list"`、`supported_in_api` 为 `true`。
这里的版本参数用于检查目录格式，不会升级本机 Codex。

发送一条实际请求：

```bash
curl --fail-with-body --silent --show-error \
  "$LITELLM_BASE_URL/responses" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-6-astra",
    "input": "Reply with exactly OK.",
    "reasoning": {"effort": "low"},
    "max_output_tokens": 64
  }'
```

正常响应的 `status` 为 `"completed"`，文本位于 `output` 中。
该请求会调用模型并使用 Copilot 配额。

## 6. 启动使用

在你的代码项目目录中运行：

```bash
codex
```

如果 Codex 已经在运行，先退出再启动，并创建新会话，不要直接恢复旧会话。
确认当前模型为 `gpt-6-astra`。如果使用 `--profile` 或 `-m` 启动，
还要确认这些参数没有覆盖上述模型和 provider 配置。

代理会自动提供以下配置，不需要在客户端重复填写：

| 项目 | GPT-6 Astra 配置 |
|------|-----------------|
| API | Responses API，`/responses` |
| 原始上下文 | 1,000,000 tokens |
| 上游最大输入 / 输出 | 872,000 / 128,000 tokens |
| Codex 有效上下文 | 870,000 tokens |
| 自动压缩阈值 | 850,000 tokens |
| 推理级别 | `low`、`medium`、`high`、`xhigh`、`max` |

默认使用 `medium`。需要更强推理时，可修改 `model_reasoning_effort`；
`max` 需要 Codex `0.144.0` 或更新版本。本代理不向 Astra 提供 `ultra` 选项。

选择 GPT-6.1 Sol 时，代理提供 `1,050,000` tokens 原始上下文、
`922,000` / `128,000` tokens 上游最大输入 / 输出、`913,500` tokens
Codex 有效上下文，以及 `900,000` tokens 自动压缩阈值。默认使用 `medium`，
Codex 推理级别与上表一致，不提供 `ultra`；直接调用 Responses API 时还支持 `none`。

## 常见问题

| 现象 | 处理方式 |
|------|----------|
| 连接被拒绝或超时 | 确认 SSH 隧道仍在运行，且配置使用 `127.0.0.1:14000`；直连时检查服务端安全组、防火墙和地址 |
| `401 Unauthorized` | 检查是否使用正确的 LiteLLM 密钥，以及启动 Codex 的终端是否设置了 `LITELLM_API_KEY` |
| 获取 token 的命令失败 | 在同一终端执行 `test -n "$LITELLM_API_KEY" && echo "key is set"`；使用本文支持的 shell 环境，确保 `printenv` 可用 |
| 找不到 `gpt-6-astra` | 确认连到更新后的代理；虚拟密钥需要有该模型的访问权限 |
| `/chat/completions` 调用失败 | Astra 仅支持 Responses API，确认 `wire_api = "responses"` |
| 上下文仍显示旧值 | 移除本地上下文和模型目录覆盖项，退出 Codex 后创建新会话 |
| 启动后还是其他模型或要求 ChatGPT 登录 | 检查用户级配置路径、`model_provider`、启动参数和所选 profile，不要只写进项目级 `.codex/config.toml` |

本文针对 Codex CLI；其他客户端需要单独确认是否支持自定义 Base URL 和
Responses API，不要假定它们会读取 Codex 的配置文件。

参考：[Codex 配置文档](https://developers.openai.com/codex/config-reference/)。
