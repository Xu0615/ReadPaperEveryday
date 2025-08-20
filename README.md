# ReadPaper (If you like it, welcome to Star~✨ )

> 这是一个用GPT协助生成的论文读取工具~每天爬取Arxiv关键词论文，然后发给LLM（可以使用gpt5-mini, gpt4o-mini等api）进行解读，帮助你跟上最新的科研进展哦~~
> **每天自动从 arXiv（计算机科学分类）抓取你关心的关键词最新论文**，解析**标题 / 作者 / 摘要 / 提交时间**，并用大模型生成「**模型解读**」。
> 支持**时间范围筛选**、**自动翻页**、**增量去重**、**同日合并**、**无新增不覆盖**，可一键用 **cron + tmux** 实现**每天 10:00 （可以自己设置）定时运行**。

------

## ✨ 功能亮点

- **按 UI 工作流抓取**：复刻 `https://arxiv.org/search/cs` 页面逻辑，按 `announced_date_first` 倒序，遇到早于 `since` 的论文即停止翻页。
- **稳健网络**：`requests` + 指数退避重试，自动处理 `5xx/429/超时`。
- **详情解析**：逐篇请求 arXiv 论文页，解析标题、作者、摘要、提交时间。
- **异步大模型解读**：`AsyncOpenAI` 并发总结（默认模型 `gpt-5-mini`，已适配部分代理参数差异，并自动兼容不支持 `temperature` 等情况）。
- **增量去重（state）**：每个关键词维护已收录的 arXiv ID 集合；**只对新增论文总结**。
- **同日合并**：同一天多次运行，新论文**合并**到当日文件，**旧内容不丢失**。
- **无新增不覆盖**：同日再次运行若**没有新增论文**，当日文件**保持不变**，仅日志提示“没有最新的论文发表”。
- **多关键词**：`KEYWORDS` 支持英文逗号分隔，逐个执行完整流程。
- **可定时**：内置 `run_daily.sh`，可配合 `cron + tmux` 每天 10:00 自动执行。

------

## 📁 目录结构（运行后）

```
readpaper/
├─ arxiv_agent.py          # 核心抓取 + 解析 + 总结 + 增量逻辑（本仓库主程序）
├─ run_daily.sh            # 每日运行脚本（读取 config.env、创建输出目录、写日志）
├─ config.env              # 你的关键词、since、API Key 等配置
├─ requirements.txt        # Python 依赖清单
├─ output/
│  └─ YYYYMMDD/
│     ├─ <SAFE_KEYWORD>_YYYYMMDD.md    # 当日 Markdown 报告（同日多次运行会“合并新增”，不丢旧内容）
│     ├─ <SAFE_KEYWORD>_YYYYMMDD.json  # 当日 JSON（结构化数据 & 模型解读）
│     └─ INDEX.json                    # 本次运行产生的文件索引
├─ state/
│  └─ seen_<SAFE_KEYWORD>.json         # 每个关键词已收录的 arXiv ID 集合（增量去重依赖）
└─ logs/
   ├─ YYYYMMDD.log                     # 本日运行日志
   └─ cron.log                         # cron 触发时的汇总日志（可选）
```

> `SAFE_KEYWORD` 为将关键词做了安全化处理后的文件名（去除空格和特殊符号）。

------

## 🚀 快速开始

### 1) 创建 Conda 环境 & 安装依赖

```
# 1) 创建 conda 环境（Python 3.11）
conda create -n readpaper python=3.11 -y

# 2) 激活环境
conda activate readpaper

# 3) 进入项目目录
cd ~/readpaper

# 4) 安装依赖
pip install -r requirements.txt
```

### 2) 配置 `config.env`

打开 `~/readpaper/config.env`，示例（已为你写好）：

```
# 关键词：多个请用英文逗号分隔；脚本会“分别”完整跑一遍流程
KEYWORDS="SPARSE AUTOENCODER"

# 只搜这个日期（含）之后提交的论文（YYYY-MM-DD）
SINCE="2025-08-01"

# 抓取节奏与翻页
SLEEP="2.0"        # 每个请求间隔秒，礼貌起见不要太小
PAGE_SIZE="50"     # 和网页一致，建议 50
MAX_PAGES="10"     # 最多翻多少页（每页 page_size 篇）

# 是否跳过模型总结（留空表示不跳过；非空则会加 --no-summarize）
# NO_SUMMARIZE="1"

# 模型 API Key（留空则不会请求模型，只收集结构化数据）
# 也可用 OPENAI_API_KEY 环境变量；这里优先使用此变量
ARXIV_AGENT_API_KEY="你的API_KEY"

# 如你的 conda 初始化脚本不在默认位置，请修改
CONDA_PROFILE="${HOME}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="readpaper"
```



### 3) 手动运行一次（建议先试跑确认）

```
# 仍在 conda 环境 (readpaper) 下
bash ~/readpaper/run_daily.sh
```

- 首次当天运行：会在 `output/YYYYMMDD/` 下生成 `<SAFE_KEYWORD>_YYYYMMDD.md/json`。
- 同一日第二次运行：
  - 若 **有新增论文** → 合并到当日文件（旧内容保留，整体按「提交时间」倒序）；
  - 若 **无新增** → **不覆盖**当日文件，仅日志提示“没有最新的论文发表”。

------



## ⏰ 每天 10:00 自动运行（cron + tmux）

> 推荐用 `cron` 触发一个小脚本 `schedule_with_tmux.sh`，让任务在 tmux 会话里运行，**无需保持 SSH 连接**。

### 0) 安装 tmux（如未安装）

```
sudo apt-get update && sudo apt-get install -y tmux
```

### 1) 创建 `schedule_with_tmux.sh`

路径：`~/readpaper/schedule_with_tmux.sh`

```
cat > ~/readpaper/schedule_with_tmux.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="readpaper_daily"

/usr/bin/tmux has-session -t "${SESSION}" 2>/dev/null && /usr/bin/tmux kill-session -t "${SESSION}"

/usr/bin/tmux new-session -d -s "${SESSION}" "/bin/bash -lc 'cd \"${PROJECT_DIR}\" && bash \"${PROJECT_DIR}/run_daily.sh\"'"
BASH

chmod +x ~/readpaper/schedule_with_tmux.sh
```

> 这里显式使用 `/usr/bin/tmux`，避免 cron 的 `PATH` 差异导致找不到可执行文件。

### 2) 配置 crontab（每天 10:00 触发）

```
crontab -e
```

在打开的编辑器底部追加以下内容：

```
SHELL=/bin/bash
CRON_TZ=Asia/Hong_Kong
PATH=~/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 每天 10:00 通过 tmux 启动任务
0 10 * * * /bin/bash -lc '~/readpaper/schedule_with_tmux.sh >> ~/readpaper/logs/cron.log 2>&1'
```

> - 若系统时区即为 HKT，可不设置 `CRON_TZ`。
> - `cron.log` 会记录每次触发；任务详细过程写在 `logs/YYYYMMDD.log`。
> - 手动随时运行：`~/readpaper/schedule_with_tmux.sh`

------

## 🧠 模型与兼容性说明

- **默认模型**：`gpt-5-mini`（见 `arxiv_agent.py` 顶部 `MODEL_NAME`）。你可改为 `gpt-4o-mini` 等。
- **代理适配**：已自动处理常见代理差异：
  - 某些代理不支持 `max_tokens` → 自动回退到 `max_completion_tokens` / `max_output_tokens`；
  - 某些模型不支持自定义 `temperature` → 自动移除该参数重试；
  - 若收到 `not_authorized_invalid_key_type` → 说明你的 Key 权限不符合该模型/项目策略，请在服务商控制台更换**有权限的 Project Key** 或放开 User Key 访问。
- **空结果**：若模型返回空文本，文档会标注 `[Empty model response]`（可换模型或复跑）。

- **增量去重**：`state/seen_<SAFE_KEYWORD>.json` 存储已收录的 `arxiv_id`，再次运行仅处理**新增论文**。
- **同日合并**：当日文件存在时，新增论文会与已存在内容合并去重，并按提交时间**倒序**重写当日文件。
- **无新增不覆盖**：同一天再次运行如无新增论文，**不覆盖**当日文件，仅日志提示“没有最新的论文发表”。
- **强制重跑/重置**：删除对应关键词的 `state/seen_<SAFE_KEYWORD>.json` 后再运行，即视作首次运行（会对 `since` 之后的所有论文重新处理）。谨慎操作。

------

## 🛠️ 故障排查（简）

- **401 not_authorized_invalid_key_type**
  你的 Key 无法访问该模型/项目：请换用**有权限的 Project Key**或在控制台允许 User Key。
- **Unsupported parameter 'max_tokens'**
  代理不支持该参数名：本程序会自动回退为 `max_completion_tokens` 或 `max_output_tokens`，保持最新代码即可。
- **Unsupported value 'temperature'**
  模型不支持自定义温度：代码已自动去掉该参数重试。
- **arXiv 429/5xx/超时**
  程序会多次重试并指数退避；若持续失败，适度调大 `SLEEP` 或稍后再试。
- **今天无新增但文件被覆盖**
  本版已修复：**无新增不覆盖**（仅第一次当日生成时会写出“没有最新的论文发表”）。

------

## 🧩 用户需要自定义的内容（务必检查）

1. **关键词与时间范围**（`config.env`）
   - `KEYWORDS="SPARSE AUTOENCODER"`（多个请用英文逗号分隔, 建议单个关键词）
   - `SINCE="2025-08-01"`（只抓该日期及之后的论文）
   - `SLEEP` / `PAGE_SIZE=50` / `MAX_PAGES`（请求节奏与翻页上限）
2. **大模型 API Key 与 Base URL**
   - `ARXIV_AGENT_API_KEY="你的API_KEY"`（或设置 `OPENAI_API_KEY` 环境变量）
   - 若需指定代理地址：设置 `OPENAI_BASE_URL` 或 `ARXIV_AGENT_BASE_URL` 环境变量（否则使用代码默认值）。
3. **模型选择（可选）**
   - 修改 `arxiv_agent.py` 顶部 `MODEL_NAME`（如 `gpt-4o-mini`）；
4. **Conda 配置**
   - `CONDA_PROFILE` 与 `CONDA_ENV_NAME`（如你的 Conda 安装路径不同）。
5. **定时任务（可选）**
   - `schedule_with_tmux.sh` 中 `/usr/bin/tmux` 路径是否匹配你的系统；
   - `crontab` 中的时区（`CRON_TZ`）与 `PATH`（确保能找到 `conda` 和 `tmux`）；
   - 触发时间（例：`0 10 * * *` → 每天 10:00）。
6. **强制全量重跑（慎用）**
   - 删除 `state/seen_<SAFE_KEYWORD>.json` 后，再运行会重新处理 `since` 之后的所有论文。

------

祝使用顺利！如需扩展（例如支持多学科、导出 CSV、Webhook 通知、企业 IM 推送等），可以在此基础上继续开发集成。
