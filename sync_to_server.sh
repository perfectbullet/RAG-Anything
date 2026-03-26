#!/bin/bash
# RAG-Anything 同步脚本
# 用法: ./sync_to_server.sh [选项]

# 配置
SERVER_HOST="zenking@192.168.8.233"
SERVER_PATH="/data/metahuman_work/RAG-Anything"
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 选项
DRY_RUN=false
VERBOSE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            DRY_RUN=true
            echo -e "${YELLOW}=== DRY RUN 模式: 不会实际同步 ===${NC}\n"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -n, --dry-run    预览模式，不实际同步"
            echo "  -v, --verbose    显示详细输出"
            echo "  -h, --help       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0              # 正常同步"
            echo "  $0 -n           # 预览将要同步的文件"
            echo "  $0 -v           # 显示详细同步信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 -h 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== RAG-Anything 同步到服务器 ===${NC}\n"
echo "本地路径: $LOCAL_PATH"
echo "服务器:   $SERVER_HOST:$SERVER_PATH"
echo ""

# 构建 rsync 命令
RSYNC_CMD="rsync -avz"

if [ "$DRY_RUN" = true ]; then
    RSYNC_CMD="$RSYNC_CMD --dry-run"
fi

if [ "$VERBOSE" = false ]; then
    RSYNC_CMD="$RSYNC_CMD --info=name0"
fi

# 排除规则
EXCLUDES=(
    # Git 和版本控制
    '.github/'
    '.git/'
    '.gitignore'

    # IDE 和编辑器
    '.vscode/'
    '.idea/'
    '.cursor/'

    # Python 缓存
    '__pycache__/'
    '*.py[cod]'
    '*.egg-info/'
    '.eggs/'
    '.pytest_cache/'
    '.mypy_cache/'
    '.ruff_cache/'

    # 虚拟环境
    'venv/'
    '.venv/'
    'env/'

    # 构建产物
    'dist/'
    'build/'
    'site/'

    # 日志文件
    '*.log'
    '*.log.*'
    'log/'

    # 环境变量和敏感文件
    '.env'
    '.env.*'
    '*.env'
    'env.example'

    # 数据和存储目录
    'rag_storage*/'
    'rag_quick_storage/'
    'rag_mvp_storage/'
    'math_kb*/'

    # 输入输出目录
    'inputs/'
    'output*/'
    'examples/input/'
    'examples/output/'
    'examples/archive'

    # 缓存目录
    '.cache/'
    '.gradio/'
    '.history/'
    'temp/'
    'tiktoken_cache/'

    # AI 助手配置
    '.claude/'
    'memory-bank/'

    # IDE 特定文件
    '.DS_Store'
    '*.swp'
    '*.swo'
    '*~'

    # 其他不需要的文件
    'local_neo4jWorkDir/'
    'neo4jWorkDir/'
    'lightrag-dev/'
    'gui/'
    'dickens*/'
    'book.txt'
    'LightRAG*.pdf'
    'download_models_hf.py'
    'TODO.md'
)

# 添加排除规则
for exclude in "${EXCLUDES[@]}"; do
    RSYNC_CMD="$RSYNC_CMD --exclude='$exclude'"
done

# 完整命令
RSYNC_CMD="$RSYNC_CMD $LOCAL_PATH/ $SERVER_HOST:$SERVER_PATH/"

echo -e "${YELLOW}执行命令:${NC}"
echo "$RSYNC_CMD"
echo ""
echo -e "${GREEN}开始同步...${NC}\n"

# 执行同步
eval $RSYNC_CMD

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}=== DRY RUN 完成 ===${NC}"
    else
        echo -e "${GREEN}=== 同步成功! ===${NC}"
    fi
else
    echo -e "${RED}=== 同步失败 (退出码: $EXIT_CODE) ===${NC}"
    exit $EXIT_CODE
fi
