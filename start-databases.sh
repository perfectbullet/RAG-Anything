#!/bin/bash

# RAGAnything 数据库服务管理脚本

set -e

echo "=========================================="
echo "  RAGAnything 数据库服务管理"
echo "=========================================="
echo ""

# 检查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检测 docker-compose 或 docker compose
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "❌ 无法找到 Docker Compose 命令"
    exit 1
fi

echo "📦 使用命令: $DOCKER_COMPOSE_CMD"
echo ""
echo "💡 数据库部署架构："
echo "   ┌────────────────────────────────────────────┐"
echo "   │ 远程服务器 (192.168.8.233)                 │"
echo "   │   - MongoDB                                │"
echo "   └────────────────────────────────────────────┘"
echo "   ┌────────────────────────────────────────────┐"
echo "   │ 本地 Docker (docker-compose)               │"
echo "   │   - Neo4j (知识图谱)                       │"
echo "   │   - Milvus (向量检索)                      │"
echo "   │   - etcd (Milvus 依赖)                     │"
echo "   │   - MinIO (Milvus 对象存储)                │"
echo "   └────────────────────────────────────────────┘"
echo ""

# 函数：启动服务
start_services() {
    echo "🚀 启动所有数据库服务..."
    $DOCKER_COMPOSE_CMD -f docker-compose.yml up -d
    echo ""
    echo "✅ 服务已启动！"
    echo ""
    show_status
    show_connection_info
}

# 函数：停止服务
stop_services() {
    echo "🛑 停止所有数据库服务..."
    $DOCKER_COMPOSE_CMD -f docker-compose.yml down
    echo ""
    echo "✅ 服务已停止！"
}

# 函数：重启服务
restart_services() {
    echo "🔄 重启所有数据库服务..."
    $DOCKER_COMPOSE_CMD -f docker-compose.yml restart
    echo ""
    echo "✅ 服务已重启！"
    echo ""
    show_status
}

# 函数：查看状态
show_status() {
    echo "📊 服务状态："
    echo ""
    $DOCKER_COMPOSE_CMD -f docker-compose.yml ps
    echo ""
}

# 函数：查看日志
show_logs() {
    if [ -z "$1" ]; then
        echo "📋 显示所有服务日志 (Ctrl+C 退出)..."
        $DOCKER_COMPOSE_CMD -f docker-compose.yml logs -f
    else
        echo "📋 显示 $1 服务日志 (Ctrl+C 退出)..."
        $DOCKER_COMPOSE_CMD -f docker-compose.yml logs -f "$1"
    fi
}

# 函数：清理数据（危险操作）
clean_data() {
    echo "⚠️  警告：此操作将删除所有数据库数据！"
    echo "⚠️  包括 MongoDB、Neo4j、Milvus 的所有数据！"
    echo ""
    read -p "确认删除所有数据？(输入 'yes' 继续): " confirm
    if [ "$confirm" = "yes" ]; then
        echo "🗑️  停止并删除所有容器和数据卷..."
        $DOCKER_COMPOSE_CMD -f docker-compose.yml down -v
        echo ""
        echo "✅ 数据已清理！"
    else
        echo "❌ 操作已取消"
    fi
}

# 函数：显示连接信息
show_connection_info() {
    echo "📌 数据库连接信息："
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ 远程数据库服务器：192.168.8.233                          │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ MongoDB (远程)                                           │"
    echo "│   URI:          mongodb://zenking:***@192.168.8.233:27017/│"
    echo "│   Database:     rag_db                                  │"
    echo "│   用户名:       zenking                                 │"
    echo "│   密码:         chanjing2025                            │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ Neo4j (本地 Docker)                                      │"
    echo "│   URI:          bolt://192.168.8.233 :7687                   │"
    echo "│   Username:     neo4j                                   │"
    echo "│   Password:     chanjing2025                           │"
    echo "│   HTTP:         http://192.168.8.233 :7474                   │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ Milvus (本地 Docker)                                     │"
    echo "│   URI:          http://192.168.8.233 :19530                  │"
    echo "│   Database:     rag_db                                  │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ MinIO (本地 Docker, Milvus 内部使用)                     │"
    echo "│   API:          http://192.168.8.233 :9000                   │"
    echo "│   Console:      http://192.168.8.233 :9001                   │"
    echo "│   Access Key:   minioadmin                              │"
    echo "│   Secret Key:   chanjing2025                           │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""
}

# 函数：测试连接
test_connections() {
    echo "🔍 测试数据库连接..."
    echo ""

    # 测试外部 MongoDB
    echo "📡 测试外部 MongoDB (192.168.8.233:27017)..."
    if mongosh "mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db" --eval "db.adminCommand('ping')" &> /dev/null; then
        echo "   ✅ MongoDB 连接成功"
    else
        echo "   ⚠️  MongoDB 连接失败（请确保："
        echo "      1. 数据库服务器 192.168.8.233 可访问"
        echo "      2. MongoDB 服务正在运行"
        echo "      3. 用户 zenking 已创建"
    fi

    # 测试本地 Neo4j
    echo "📡 测试本地 Neo4j (192.168.8.233 :7687)..."
    if cypher-shell -a "bolt://neo4j:chanjing2025@192.168.8.233 :7687" "RETURN 1" &> /dev/null; then
        echo "   ✅ Neo4j 连接成功"
    else
        echo "   ❌ Neo4j 连接失败（请先运行: ./start-databases.sh start）"
    fi

    # 测试本地 Milvus
    echo "📡 测试本地 Milvus (192.168.8.233 :19530)..."
    if curl -s http://192.168.8.233 :19530/healthz &> /dev/null; then
        echo "   ✅ Milvus 连接成功"
    else
        echo "   ❌ Milvus 连接失败（请先运行: ./start-databases.sh start）"
    fi

    # 测试本地 MinIO
    echo "📡 测试本地 MinIO (192.168.8.233 :9000)..."
    if curl -s http://192.168.8.233 :9000/minio/health/live &> /dev/null; then
        echo "   ✅ MinIO 连接成功"
    else
        echo "   ❌ MinIO 连接失败（请先运行: ./start-databases.sh start）"
    fi

    echo ""
}

# 主菜单
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    clean)
        clean_data
        ;;
    test)
        test_connections
        ;;
    info)
        show_connection_info
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs|clean|test|info}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动所有数据库服务"
        echo "  stop    - 停止所有数据库服务"
        echo "  restart - 重启所有数据库服务"
        echo "  status  - 查看服务状态"
        echo "  logs    - 查看日志 (可指定服务名: $0 logs mongodb)"
        echo "  clean   - 清理所有数据（危险操作）"
        echo "  test    - 测试数据库连接"
        echo "  info    - 显示数据库连接信息"
        echo ""
        exit 1
        ;;
esac
