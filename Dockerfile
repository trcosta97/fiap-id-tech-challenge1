FROM python:3.12-slim

# Evita prompts interativos durante instalação de pacotes do sistema
ENV DEBIAN_FRONTEND=noninteractive

# Diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para algumas libs (matplotlib, shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (aproveita cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Porta padrão do JupyterLab
EXPOSE 8888

# Inicia JupyterLab sem token (acesso local via docker-compose)
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
