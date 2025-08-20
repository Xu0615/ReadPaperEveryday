#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_DIR}/config.env"

TODAY="$(date +%Y%m%d)"
OUTDIR="${PROJECT_DIR}/output/${TODAY}"
LOGDIR="${PROJECT_DIR}/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

# 实时输出
export PYTHONUNBUFFERED=1

# 从 config.env 读到的 ARXIV_AGENT_API_KEY 同步到两个变量名，防止某些环境只识别其中一个
export ARXIV_AGENT_API_KEY="${ARXIV_AGENT_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${ARXIV_AGENT_API_KEY:-}}"

# 诊断打印（不打印具体值）
AK_STATUS="EMPTY"; [[ -n "${ARXIV_AGENT_API_KEY}" ]] && AK_STATUS="SET"
OK_STATUS="EMPTY"; [[ -n "${OPENAI_API_KEY}" ]] && OK_STATUS="SET"
echo "[RUN] $(date) -- keywords='${KEYWORDS}' since='${SINCE}' outdir='${OUTDIR}'  AK=${AK_STATUS} OK=${OK_STATUS}" | tee -a "${LOGDIR}/${TODAY}.log"

NO_SUM_FLAG=""
if [ "${NO_SUMMARIZE:-}" != "" ]; then
  NO_SUM_FLAG="--no-summarize"
fi

# 固定使用你的 conda 环境解释器（无需 conda activate），比如/home/xxx/.conda/envs/${CONDA_ENV_NAME}/bin/python
PYTHON_CMD=""

# 若上面的路径不存在（比如 CONDA_ENV_NAME 不是 readpaper 或路径不同），回退到 conda run
if [ ! -x "$PYTHON_CMD" ]; then
  PYTHON_CMD="/home/xxx/miniconda3/bin/conda run -n ${CONDA_ENV_NAME} python"
fi

# 记录实际使用解释器与 httpx 可用性，便于在 cron 日志诊断
"$PYTHON_CMD" - <<'PY'
import sys, pkgutil
print(f"[ENV] sys.executable={sys.executable}")
print(f"[ENV] httpx_installed={bool(pkgutil.find_loader('httpx'))}")
PY


set -x
${PYTHON_CMD} "${PROJECT_DIR}/arxiv_agent.py" \
  --keywords "${KEYWORDS}" \
  --since "${SINCE}" \
  --page-size "${PAGE_SIZE:-50}" \
  --max-pages "${MAX_PAGES:-10}" \
  --sleep "${SLEEP:-2.0}" \
  --outdir "${OUTDIR}" \
  --state-dir "${PROJECT_DIR}/state" \
  ${NO_SUM_FLAG} 2>&1 | tee -a "${LOGDIR}/${TODAY}.log"
set +x

echo "[DONE] $(date)" | tee -a "${LOGDIR}/${TODAY}.log"
