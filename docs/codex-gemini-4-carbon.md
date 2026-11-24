# 在另一台机器配置 Codex（CLI / App / IDE 扩展）+ Gemini 4 Carbon / Neon

本文适用于 **macOS、Linux 以及 Windows（WSL）**。目标机器只需安装 Codex 客户端（Codex CLI、独立 Codex App 或 VS Code / Cursor 的 Codex 扩展），直接连接已部署好的 `litellm-ghc-proxy` 服务，无需在新机器上部署 Docker 或登录 GitHub Copilot / ChatGPT。

> 本文同样适用于代理支持的其他 Codex 模型（如 `gpt-6.2-sol`、`gpt-6.1-sol`、`gpt-6-astra`、`gpt-5.6-sol` 等），只需切换 `model` 与对应的 `model_reasoning_effort` 即可，也可直接在客户端模型选择器中自由切换。

当前服务端地址按项目配置为 `20.89.181.138`，监听端口为 `4000`。若服务器 IP 或域名有变化，请替换下文对应地址。

## 使用 Gemini 4 Neon

下文认证、连接与安装步骤同样适用。若已配置 Carbon，只需在
`~/.codex/config.toml` 顶层切换：

```toml
model = "gemini-4-neon"
model_reasoning_effort = "xhigh"
```

Neon 支持 `low`、`medium`（默认）、`high`、`xhigh` 四档，不同于 Carbon 的
最高 `high`。原始上下文为 `1,048,576` tokens，最大输入/输出为
`983,040` / `65,536`；代理下发的 Codex 有效窗口为 `975,175`（93%），
自动压缩阈值为 `950,000`。工具、图片及 `wire_api = "responses"` 配置不变。
使用虚拟密钥时需授权 `gemini-4-neon`；删除旧 `model_catalog_json` 覆盖，
重启客户端并新建会话后可在模型选择器中选择 **Gemini 4 Neon**。

---

## 1. 安装 Codex 客户端

在新机器按需安装 CLI 或桌面端 / IDE 扩展（共用同一份 `~/.codex/config.toml`）：

```bash
# 方式 1：npm 安装 CLI
npm install -g @openai/codex@latest

# 方式 2：macOS Homebrew 安装
brew install --cask codex

# 验证版本
codex --version
```

---

## 2. 连接代理服务器

**推荐方式：SSH 本地端口转发（加密传输，无需暴露公网 4000 端口）**

在新机器单独打开一个终端（或配置 `autossh` / `~/.ssh/config` 后台隧道），将 `<SSH用户名>` 替换为服务器登录用户名：

```bash
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:14000:127.0.0.1:4000 \
  <SSH用户名>@20.89.181.138
```

保持该隧道运行，新机器本地 `http://127.0.0.1:14000` 即映射到远程代理的 `4000` 端口。

| 连接方式 | `base_url` 填写值 | 说明 |
|----------|-------------------|------|
| **SSH 隧道（默认推荐）** | `http://127.0.0.1:14000` | 全程 SSH 加密，无需放行公网 4000 |
| **HTTPS 反向代理** | `https://<你的代理域名>` | 适合已配置域名与 TLS 证书的场景 |
| **HTTP 直连** | `http://20.89.181.138:4000` | 明文传输，仅限可信内网或安全组白名单 IP |

---

## 3. 配置代理访问密钥

访问密钥为代理服务器 `.env` 中的 `LITELLM_MASTER_KEY`（或管理员签发的、已授权访问 `gemini-4-carbon` 的 LiteLLM Virtual Key `sk-...`）。

为了让 Codex 自动从代理拉取 `/models` 模型目录（从而在 Codex App / CLI 界面中显示 **Gemini 4 Carbon** 及其 **Low / Medium / High** 三档选择器），需要使用 `[model_providers.litellm-proxy.auth]` 命令认证。请根据你的使用方式二选一：

### 方式 A（推荐：同时支持桌面端 Codex App、VS Code/Cursor 扩展与终端 CLI）

macOS / Linux 从程序坞或桌面图标启动 GUI 应用（Codex App、VS Code、Cursor）时，默认**不会加载** `~/.zshrc` / `~/.bashrc` 中的环境变量。将密钥保存在本机 `600` 权限文件中最为稳妥：

```bash
mkdir -p ~/.codex
printf '%s\n' '<替换为你的代理访问密钥>' > ~/.codex/.litellm_api_key
chmod 600 ~/.codex/.litellm_api_key

# 查看密钥文件的绝对路径（复制输出路径，下一步填入 config.toml）
realpath ~/.codex/.litellm_api_key
```

> 例如 macOS 通常为 `/Users/<用户名>/.codex/.litellm_api_key`，Linux 通常为 `/home/<用户名>/.codex/.litellm_api_key`。

### 方式 B（仅在终端运行 `codex` CLI）

将环境变量写入 `~/.zshrc` 或 `~/.bashrc` 并重新打开终端：

```bash
export LITELLM_API_KEY="<替换为你的代理访问密钥>"
```

---

## 4. 编写 `~/.codex/config.toml`

编辑新机器上的 `~/.codex/config.toml`：

### 若采用「方式 A（密钥文件，兼容 Codex App + CLI）」：

```toml
model = "gemini-4-carbon"
model_provider = "litellm-proxy"
model_reasoning_effort = "high"

[model_providers.litellm-proxy]
name = "LiteLLM GitHub Copilot Proxy"
base_url = "http://127.0.0.1:14000"
wire_api = "responses"
requires_openai_auth = false

[model_providers.litellm-proxy.auth]
command = "cat"
args = ["/Users/<你的用户名>/.codex/.litellm_api_key"]
```

*(Linux 请将 `args` 中的路径改为 `/home/<你的用户名>/.codex/.litellm_api_key`；若使用 HTTP 直连，将 `base_url` 改为 `"http://20.89.181.138:4000"`。)*

### 若采用「方式 B（环境变量，仅终端 CLI）」：

将最后一节 `[model_providers.litellm-proxy.auth]` 改为：

```toml
[model_providers.litellm-proxy.auth]
command = "printenv"
args = ["LITELLM_API_KEY"]
```

### 配置注意事项（重要）

1. **顶层字段位置**：`model`、`model_provider`、`model_reasoning_effort` 必须写在所有 `[...]` 表头**之前**。
2. **清理旧覆盖项**：如果新机器以前配置过以下字段，请**删除**，否则会阻止代理下发最新模型目录与上下文参数：
   - 删除 `model_catalog_json = ...`
   - 删除 `model_context_window = ...` 与 `model_auto_compact_token_limit = ...`
   - 删除 `[model_providers.litellm-proxy]` 下的旧 `env_key = ...`（与 `.auth` 二选一，保留 `.auth` 才会自动请求 `/models` 目录）
3. **三档推理强度 (`model_reasoning_effort`)**：
   - `gemini-4-carbon` 支持三档：`"low"`、`"medium"`（目录默认）、`"high"`（最高档位）。
   - 若从 GPT-5.6 / GPT-6.2 切换过来导致配置中残留 `"xhigh"` 或 `"max"`，代理中间件也会自动将其钳制为 `"high"`，不会报错中断。

---

## 5. 验证连通性与三档推理

在新机器终端执行以下两条命令，确认网络、密钥与模型目录均正常：

```bash
# 读取密钥与代理地址（以方式 A + SSH 隧道为例；直连请改 LITELLM_BASE_URL）
LITELLM_API_KEY="$(cat ~/.codex/.litellm_api_key)"
LITELLM_BASE_URL="http://127.0.0.1:14000"

# 1) 检查 Codex 模型目录是否返回 gemini-4-carbon 及三档 effort
curl --fail-with-body -sS \
  "$LITELLM_BASE_URL/models?client_version=0.144.0" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); m=next(x for x in d["models"] if x["slug"]=="gemini-4-carbon"); print(m["display_name"], [e["effort"] for e in m["supported_reasoning_levels"]], "ctx=", m["context_window"])'
```

预期输出：

```text
Gemini 4 Carbon ['low', 'medium', 'high'] ctx= 1000000
```

再测试一次实际 `/responses` 推理调用（最高档 `high`）：

```bash
# 2) 验证 Gemini 4 Carbon 实际响应
curl --fail-with-body -sS \
  "$LITELLM_BASE_URL/responses" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-4-carbon",
    "input": "Reply with exactly CARBON_READY.",
    "reasoning": {"effort": "high"}
  }'
```

返回的 JSON 中 `"status": "completed"` 且 `output` 包含 `"CARBON_READY"` 即表示全链路就绪。

---

## 6. 启动使用

- **Codex CLI**：进入任意项目目录执行 `codex`（或单次执行 `codex exec "你的任务"`），输入 `/model` 可随时切换模型与 `Low / Medium / High` 三档推理强度。
- **Codex App / VS Code / Cursor Codex 扩展 / ChatGPT.app 本地模式**：如果应用已在运行，**完全退出并重启应用**，然后新建线程（New Thread），即可在模型选择器中看到 **Gemini 4 Carbon** 以及 **Low / Medium / High** 三档选项。
  > 注：若使用的是纯云端网页版 Codex 会话，它固定连接官方云端后端、不读取本机 `~/.codex/config.toml`；请使用读取本地 `~/.codex/config.toml` 的客户端（独立 Codex App、ChatGPT.app 本地 Codex 环境、VS Code / Cursor Codex 扩展或 `codex` CLI）。

### 代理自动下发的 Gemini 4 Carbon 参数一览

| 参数项 | 代理配置值 |
|--------|------------|
| 模型 ID (`model`) | `gemini-4-carbon` |
| 客户端协议 (`wire_api`) | `"responses"`（中间件自动桥接至 Copilot `/chat/completions`） |
| 原始上下文窗口 | `1,000,000` tokens |
| 上游最大输入 / 最大输出 | `936,000` / `64,000` tokens |
| Codex 有效上下文窗口 | `930,000` tokens（93%） |
| 自动压缩阈值 | `900,000` tokens |
| 支持的推理档位 | `low`、`medium`（默认）、`high`（最高） |
| 补丁工具格式 | `freeform`（兼容各版本 Codex `ApplyPatchToolType`，由中间件自动桥接 `custom_tool_call`） |

---

## 7. 常见问题排查

| 现象 | 原因与解决方法 |
|------|----------------|
| Codex App 模型列表中看不到 `Gemini 4 Carbon` | 1. 检查是否删除了旧的 `model_catalog_json` 和 `env_key`；2. 确认配置了 `[model_providers.litellm-proxy.auth]`；3. 完全退出并重启 Codex App 后新建会话 |
| Codex App 报获取认证 Token 失败（CLI 却正常） | GUI 应用未继承终端环境变量；改用本文第 3 节「方式 A」的密钥文件 + `command = "cat"`，且 `args` 填写**绝对路径**（不要用 `~`） |
| `401 Unauthorized` | 密钥填写错误，或使用的是未授予 `gemini-4-carbon` 权限的 LiteLLM 虚拟密钥（`sk-...` 需在 LiteLLM UI `/ui` 中勾选允许 `gemini-4-carbon`） |
| 连接超时 / `Connection refused` | 若使用 SSH 隧道，确认隧道进程未断开；若使用直连，确认服务器防火墙/安全组对当前机器放行了 TCP `4000` |
| `request_too_large` / HTTP 413，但 `/status` 显示上下文还有空间 | 上游请求体或附件大小限制与 930K token 窗口无关。错误会显示客户端/转发请求字节数；请缩小图片、移除大附件，或减少工具输出/历史后重试。若大内容已进入旧会话历史，可编辑移除该输入，或携带简短任务摘要新建会话；单纯调大上下文窗口或原样重试无效 |
| 启动后提示登录 ChatGPT | 检查配置文件路径是否为 `~/.codex/config.toml`，且顶层 `model_provider = "litellm-proxy"` 未被放在某个 `[...]` 表头下方 |
