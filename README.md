# FIAP ID Tech Challenge 1 — Câncer de Mama

Projeto de Machine Learning para classificação do desfecho de pacientes com câncer de mama (`Status`: Alive / Dead) usando o dataset `Breast_Cancer.csv`.

---

## Vídeo de Demonstração

[![Assista no YouTube](https://img.shields.io/badge/YouTube-Assistir-red?logo=youtube)](https://youtu.be/7Bw4rorYXZs)

---

## Notebooks

| Arquivo | Descrição |
|---|---|
| `eda_breast_cancer.ipynb` | Análise Exploratória de Dados (EDA) |
| `ml_breast_cancer.ipynb` | Treinamento e avaliação dos modelos |

---

## Como rodar com Docker

### Pré-requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando

### Subir o ambiente

```bash
docker compose up --build
```

Na primeira execução o build pode levar alguns minutos (instalação das dependências). Nas próximas vezes use apenas:

```bash
docker compose up
```

### Acessar o JupyterLab

Abra no navegador: [http://localhost:8888](http://localhost:8888)

Não é necessário token ou senha.

### Parar o container

```bash
docker compose down
```

---

## Estrutura do projeto

```
.
├── base/
│   └── Breast_Cancer.csv       # Dataset
├── eda_breast_cancer.ipynb     # Notebook de EDA
├── ml_breast_cancer.ipynb      # Notebook de ML
├── requirements.txt            # Dependências Python
├── Dockerfile                  # Imagem do container
├── docker-compose.yml          # Orquestração do container
└── .dockerignore               # Arquivos ignorados no build
```

---

## Dataset

**Fonte:** [Breast Cancer Dataset — Kaggle](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer)

- **4.024 amostras** × 16 colunas
- **Variável-alvo:** `Status` (Alive / Dead)
- **Distribuição:** ~84,7% Alive / ~15,3% Dead (desbalanceamento de classe)
- **Sem valores ausentes**

### Features utilizadas

| Tipo | Variáveis |
|---|---|
| Numéricas (4) | Age, Tumor Size, Regional Node Examined, Reginol Node Positive |
| Categóricas (10) | Race, Marital Status, T Stage, N Stage, 6th Stage, differentiate, Grade, A Stage, Estrogen Status, Progesterone Status |

> `Survival Months` foi **removida** para evitar *data leakage* — trata-se de dado prospectivo que não existe no momento do diagnóstico.

---

## Análise Exploratória de Dados (EDA)

### Principais achados

**Desbalanceamento de classe**
O dataset apresenta ~85% de pacientes Alive e ~15% Dead. Qualquer modelo de classificação precisa tratar esse desbalanceamento (via `class_weight`, `scale_pos_weight` ou SMOTE) para evitar predições enviesadas para a classe majoritária.

**Variáveis mais discriminativas**
As variáveis numéricas `Tumor Size`, `Regional Node Examined` e `Reginol Node Positive` apresentam as maiores correlações mútuas e a separação mais clara entre os grupos de sobrevivência nos boxplots, sendo os prováveis preditores de maior relevância.

**Estadiamento clínico (T Stage / N Stage)**
Estágios mais avançados correspondem a tumores maiores e maior envolvimento de linfonodos. Pacientes Dead estão concentrados nos estágios superiores, confirmando o comportamento oncológico esperado.

**Receptores hormonais (Estrogen Status / Progesterone Status)**
Pacientes com receptores positivos apresentam taxas de mortalidade consistentemente menores, alinhado com a evidência clínica estabelecida — esses pacientes têm acesso à hormonoterapia.

**Meses de Sobrevivência**
As distribuições de `Survival Months` são marcadamente distintas entre os grupos Alive e Dead, o que confirma seu alto valor preditivo — mas reforça o motivo de sua remoção (data leakage).

**Qualidade dos dados**
Nenhum valor ausente foi detectado. Algumas variáveis numéricas apresentam distribuições assimétricas à direita e outliers consideráveis (especialmente contagens de linfonodos), mas esses valores extremos refletem casos clínicos reais, não erros de coleta.

---

## Pré-processamento

1. **Remoção de `Survival Months`** — dado prospectivo, não disponível no momento do diagnóstico
2. **Encode do alvo** — `LabelEncoder`: Alive = 0, Dead = 1 (Dead é a classe positiva/de risco)
3. **Variáveis categóricas** — `OrdinalEncoder` com `handle_unknown='use_encoded_value'` (evita erros em dados não vistos)
4. **Variáveis numéricas** — `StandardScaler`
5. **Pipeline `ColumnTransformer`** — aplica as transformações de forma encadeada e reproduzível
6. **Divisão treino/teste** — 80/20 estratificada (`random_state=42`), mantendo as proporções de classe em ambos os conjuntos

---

## Modelagem

Foram avaliados três modelos de classificação. O critério de seleção foi o **F1-Score e ROC-AUC na classe Dead (positiva)**, pois em diagnóstico oncológico minimizar falsos negativos (predizer Alive para quem vai morrer) é a prioridade.

### Modelos

**Regressão Logística**
Baseline interpretável para classificação binária. `class_weight='balanced'` compensa o desbalanceamento de classe.

**Random Forest**
Ensemble de árvores de decisão, não-linear e robusto a outliers e dados mistos. `class_weight='balanced_subsample'` aplica balanceamento por árvore.

**XGBoost**
Gradient boosting com regularização L1/L2. `scale_pos_weight=5.53` (razão Alive/Dead no treino) compensa o desbalanceamento.

### Resultados

| Modelo | Accuracy | Precision (Dead) | Recall (Dead) | F1 (Dead) | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| **Regressão Logística** | 0,700 | 0,277 | **0,594** | **0,377** | **0,719** | 0,411 ± 0,018 |
| Random Forest | 0,826 | 0,384 | 0,228 | 0,286 | 0,697 | 0,332 ± 0,029 |
| XGBoost | 0,767 | 0,290 | 0,366 | 0,324 | 0,677 | 0,336 ± 0,027 |

**Melhor modelo: Regressão Logística**
Apesar da menor accuracy, apresentou o maior F1-Score (0,377), maior ROC-AUC (0,719) e maior Recall (0,594) para a classe Dead — detectando ~59% dos casos reais de óbito, o que é a métrica mais relevante para este problema clínico.

---

## Explicabilidade

### Feature Importance
Para a Regressão Logística, a importância das features foi calculada a partir dos valores absolutos dos coeficientes, servindo como proxy de relevância. As variáveis de receptores hormonais (`Progesterone Status`, `Estrogen Status`), estadiamento (`N Stage`, `6th Stage`) e envolvimento nodal (`Reginol Node Positive`) foram as mais impactantes.

### SHAP (SHapley Additive exPlanations)
O SHAP foi aplicado ao melhor modelo para quantificar a contribuição de cada feature em cada predição individual:

- **Progesterone Status / Estrogen Status positivos** → reduzem a probabilidade de Dead (SHAP negativo)
- **Reginol Node Positive alto** → aumenta a probabilidade de Dead (SHAP positivo)
- Os resultados são coerentes com o conhecimento clínico estabelecido

---

## Discussão Crítica

### O modelo pode ser utilizado na prática?

O modelo pode ser utilizado como **ferramenta de suporte à decisão** em triagem oncológica — mas nunca como substituto do julgamento clínico. O médico deve ter sempre a palavra final.

**Limitações identificadas:**
- Recall de 59% para a classe Dead significa que ~41% dos casos de óbito ainda não são detectados
- O desbalanceamento (~85/15) foi mitigado mas não eliminado — técnicas como SMOTE podem melhorar os resultados
- Os dados são de um único dataset público; validação em dados hospitalares externos é necessária antes de qualquer uso clínico

---

## Dependências principais

- Python 3.12
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, shap
- JupyterLab
