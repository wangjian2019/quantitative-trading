# AI量化交易平台 Docker配置
# Author: Alvin
# 支持Java + Python混合架构

FROM openjdk:8-jdk-slim

# 安装Python和必要工具
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Maven
RUN wget https://archive.apache.org/dist/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz \
    && tar -xzf apache-maven-3.8.6-bin.tar.gz -C /opt \
    && ln -s /opt/apache-maven-3.8.6 /opt/maven \
    && rm apache-maven-3.8.6-bin.tar.gz

ENV MAVEN_HOME=/opt/maven
ENV PATH=$PATH:$MAVEN_HOME/bin

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN cd strategy && pip3 install -r requirements.txt

# 编译Java项目
RUN cd platform && mvn clean compile

# 创建日志目录
RUN mkdir -p logs strategy/logs strategy/models

# 暴露端口
EXPOSE 5000 8080

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "启动AI量化交易平台 Docker容器"\n\
echo "Author: Alvin"\n\
echo "================================"\n\
\n\
# 启动Python AI服务（后台运行）\n\
cd /app/strategy\n\
echo "启动Python AI服务..."\n\
python3 ai_strategy_service.py &\n\
AI_PID=$!\n\
\n\
# 等待AI服务启动\n\
sleep 10\n\
\n\
# 启动Java交易平台\n\
cd /app/platform\n\
echo "启动Java交易平台..."\n\
mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformMain"\n\
\n\
# 清理\n\
kill $AI_PID 2>/dev/null || true\n\
' > /app/start_docker.sh && chmod +x /app/start_docker.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# 启动命令
CMD ["/app/start_docker.sh"]
