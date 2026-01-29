# 🚀 Passos Mágicos: Sistema Preditivo de Risco de Evasão (MLOps Ready)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-F7931E.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-0.4.1-6F42C1.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MCA1MCI+PHBhdGggZmlsbD0iIzY0NTRiYSIgZD0iTTI1IDBDMTEuMTkgMCAwIDExLjE5IDAgMjVzMTEuMTkgMjUgMjUgMjUgMjUtMTEuMTkgMjUtMjVTMzguODEgMCAyNSAwem0wIDQ1Yy0xMS4wMyAwLTIwLTguOTctMjAtMjBzOC45Ny0yMCAyMC0yMCAyMCA4Ljk3IDIwIDIwLTguOTcgMjAtMjAgMjB6Ii8+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTI1IDVjLTExLjA1IDAtMjAgOC45NS0yMCAyMHM4Ljk1IDIwIDIwIDIwIDIwLTguOTUgMjAtMjBzLTguOTUtMjAtMjAtMjB6bTAgMzVjLTguMjggMC0xNS02LjcyLTE1LTE1czYuNzItMTUgMTUtMTUgMTUgNi43MiAxNSAxNS02LjcyIDE1LTE1IDE1eiIvPjxwYXRoIGZpbGw9IiM2NDU0YmEiIGQ9Ik0yNSAxMGMtOC4yOCAwLTE1IDYuNzItMTUgMTVzNi43MiAxNSAxNSAxNSAxNS02LjcyIDE1LTE1cy02LjcyLTE1LTE1LTE1em0wIDI1Yy01LjUyIDAtMTAtNC40OC0xMC0xMHM0LjQ4LTEwIDEwLTEwIDEwIDQuNDggMTAgMTAtNC40OCAxMC0xMCAxMHoiLz48L3N2Zz4=)](https://evidentlyai.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10.17-2496ED.svg?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Produção%20Candidata-green.svg?style=for-the-badge)](https://github.com/PauloGuedes2/datathon-tech5/tree/feature/modelo-novo)

> **Previsão Preditiva de Risco de Evasão Escolar:** Uma arquitetura MLOps para intervenção social e educacional proativa.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [O Problema que Este Projeto Resolve](#2-o-problema-que-este-projeto-resolve)
3. [O Que Este Projeto É](#3-o-que-este-projeto-é)
4. [O Que Este Projeto NÃO É](#4-o-que-este-projeto-não-é)
5. [Arquitetura da Solução](#5-arquitetura-da-solução)
6. [Pipeline de Machine Learning](#6-pipeline-de-machine-learning)
7. [Justificativas Técnicas](#7-justificativas-técnicas)
8. [Stack Tecnológica](#8-stack-tecnológica)
9. [Estrutura do Projeto](#9-estrutura-do-projeto)
10. [API e Deployment](#10-api-e-deployment)
11. [Testes e Qualidade](#11-testes-e-qualidade)
12. [Monitoramento e Observabilidade](#12-monitoramento-e-observabilidade)
13. [Segurança e Confiabilidade](#13-segurança-e-confiabilidade)
14. [Limitações Conhecidas](#14-limitações-conhecidas)
15. [Possíveis Evoluções](#15-possíveis-evoluções)
16. [Instruções de Execução](#16-instruções-de-execução)
17. [Conformidade com o Datathon](#17-conformidade-com-o-datathon)
18. [Uso Responsável e Ético](#18-uso-responsável-e-ético)
19. [Considerações Finais](#19-considerações-finais)
20. [Licença e Contribuição](#20-licença-e-contribuição)

---

## 1. Visão Geral

Este projeto apresenta uma solução completa de **Machine Learning Operacional (MLOps)** para prever o risco de evasão escolar em alunos da ONG Passos Mágicos. O objetivo central é fornecer uma ferramenta preditiva que permita à equipe pedagógica realizar **intervenções proativas** antes que o risco se materialize.

A solução foi desenvolvida com foco em **integridade de dados**, **robustez de produção** e **prevenção de *data leakage***, utilizando uma arquitetura de microsserviços baseada em FastAPI e um pipeline de ML que simula um ambiente de produção de alto rigor técnico.

O principal diferencial técnico é a implementação de **Features Históricas (*Lag Features*)** e um **Split Temporal** rigoroso, garantindo que o modelo utilize apenas informações do passado (ano T-1) para prever o risco no presente (ano T), eliminando o risco de vazamento de dados (*leakage*) e aumentando a confiança na capacidade preditiva em cenários reais.

## 2. O Problema que Este Projeto Resolve

A evasão escolar e o baixo desempenho acadêmico são problemas complexos com alto **impacto social e econômico**. A identificação tardia de alunos em situação de risco impede a aplicação de medidas corretivas eficazes.

Este sistema resolve a necessidade crítica de **antecipação**. Ao invés de diagnosticar o risco após a ocorrência (ex: após uma nota baixa ou defasagem de idade/série), o modelo prediz a probabilidade de risco **no início do ano letivo**, utilizando o histórico do aluno. Isso transforma a intervenção de reativa para proativa, maximizando as chances de sucesso pedagógico.

## 3. O Que Este Projeto É

O projeto é um **Sistema de Inferência de Risco em Tempo Real** com um **Pipeline de Treinamento MLOps**.

| Componente | Descrição |
| :--- | :--- |
| **API de Predição** | Serviço *stateless* (sem estado) de baixa latência, pronto para ser integrado a sistemas de gestão escolar ou *dashboards* de acompanhamento. |
| **Feature Store (In-Memory)** | Repositório de dados históricos (`HistoricalRepository`) que enriquece automaticamente as requisições de predição com as métricas do ano anterior do aluno. |
| **Pipeline de Treinamento** | Rotina robusta que carrega dados, aplica engenharia de *features* (incluindo *lag features*), treina o modelo com *Quality Gate* e o promove para produção. |
| **Monitoramento de Drift** | Endpoint dedicado que expõe um *dashboard* do Evidently AI, comparando a distribuição dos dados de produção (inferência) com os dados de referência (treinamento), garantindo a validade do modelo ao longo do tempo. |

## 4. O Que Este Projeto NÃO É

É fundamental definir o escopo para gerenciar expectativas e riscos:

*   **Não é um sistema de *data warehousing***: O `HistoricalRepository` é um *Feature Store* em memória (Singleton) para enriquecimento em tempo real. Ele não substitui um banco de dados transacional ou um *data lake*.
*   **Não é um sistema de *retraining* automático (CI/CD completo)**: Embora o pipeline de treinamento (`train.py`) seja robusto, a execução do *retraining* e a orquestração (ex: Airflow, Kubeflow) não estão implementadas. O *retraining* é executado manualmente via `python app/train.py`.
*   **Não é um sistema de *backtesting* completo**: O *Quality Gate* avalia o modelo candidato em relação ao modelo atual, mas não realiza uma análise exaustiva de *backtesting* em janelas temporais múltiplas.
*   **Não possui autenticação/autorização (AuthN/AuthZ)**: A API de predição é aberta. Em um ambiente de produção real, seria obrigatório implementar um mecanismo de segurança (ex: *API Key*, OAuth2) para proteger o endpoint sensível.

## 5. Arquitetura da Solução

A arquitetura segue o padrão de **Arquitetura Hexagonal/Limpa** para desacoplamento de camadas, facilitando a manutenção e a troca de tecnologias.

```mermaid 
graph TD
    A[Usuário/Sistema Externo] -->|POST /predict/smart| B[FastAPI - app/main.py];
    B --> C{Controller de Predição};
    C --> D[RiskService - Aplicação];
    D --> E[ModelManager - Infra];
    D --> F[HistoricalRepository - Infra];
    D --> G[PredictionLogger - Infra];
    E -->|Carrega Modelo| H[model_passos_magicos.joblib];
    F -->|Busca Histórico T-1| I[Dados Históricos/Feature Store];
    D -->|Aplica FeatureProcessor| J[Dados Prontos para Predição];
    J --> E;
    E -->|Resultado| C;
    C -->|Resposta JSON| A;
    G -->|Log JSONL| K[prediction.jsonl];

    subgraph MLOps Pipeline (Offline)
        L[Execução Manual: python app/train.py] --> M[DataLoader];
        M --> N[MLPipeline];
        N --> O[Cria Lag Features];
        N --> P[Split Temporal];
        N --> Q[Treinamento c/ Quality Gate];
        Q --> H;
        Q --> R[metrics.json];
        Q --> S[reference_data.csv];
    end

    subgraph Observabilidade (Online)
        T[Usuário/DevOps] -->|GET /monitoring/dashboard| U[MonitoringController];
        U --> V[MonitoringService];
        V -->|Compara| S;
        V -->|Compara| K;
        V -->|Gera Dashboard HTML| T;
    end
```

**Componentes Chave:**

*   **`app/main.py`**: Ponto de entrada da API, responsável por inicializar o `FastAPI` e carregar o modelo em memória no *startup* (`@app.on_event("startup")`).
*   **`src/api/controller.py`**: Camada de interface, recebe requisições e utiliza o `Depends` do FastAPI para injetar o `RiskService` com o modelo já carregado.
*   **`src/application/risk_service.py`**: Camada de lógica de negócio. Orquestra a busca de histórico (`HistoricalRepository`), o processamento de *features* (`FeatureProcessor`) e a predição.
*   **`src/infrastructure/model/model_manager.py`**: Singleton thread-safe que gerencia o ciclo de vida do modelo em memória.

## 6. Pipeline de Machine Learning

O pipeline de ML foi desenhado para ser **rigorosamente preditivo** e **resistente a *data leakage***.

### 6.1. Engenharia de Features (Anti-Leakage)

A principal inovação é a criação de *Lag Features* (variáveis históricas) dentro do `MLPipeline` (`create_lag_features`).

| Feature | Descrição | Fonte de Dados |
| :--- | :--- | :--- |
| `INDE_ANTERIOR` | Índice de Desempenho Educacional do ano **T-1**. | Calculado via `groupby('RA').shift(1)` |
| `ALUNO_NOVO` | Flag booleana (1/0) que indica se o aluno não possui histórico (`INDE_ANTERIOR` é 0). | Derivado do `INDE_ANTERIOR` |
| `TEMPO_NA_ONG` | Anos desde o `ANO_INGRESSO` até o `ANO_REFERENCIA`. | Calculado via `FeatureProcessor` |

### 6.2. Estratégia de Treinamento e Validação

1.  **Criação do Target (Gabarito):** A variável alvo (`RISCO_DEFASAGEM`) é criada a partir de métricas atuais (`INDE`, `DEFASAGEM`, `PEDRA`).
2.  **Separação Temporal:** O conjunto de dados é dividido em Treino (anos T-2 e anteriores) e Teste (ano T-1). Isso simula o cenário real onde o modelo é treinado com dados antigos e avaliado em dados mais recentes, garantindo que a performance não seja inflada por *leakage* temporal.
3.  **Remoção de Vazamento:** Todas as colunas que definem o *target* no ano T (`INDE`, `NOTA_PORT`, etc.) são removidas do conjunto de *features* (`COLUNAS_PROIBIDAS_NO_TREINO`), forçando o modelo a aprender apenas com o histórico (`INDE_ANTERIOR`, etc.) e dados demográficos.
4.  **Quality Gate:** O modelo só é promovido se o seu F1-Score no conjunto de teste for **igual ou superior a 95%** do F1-Score do modelo atualmente em produção (`_should_promote_model`).

## 7. Justificativas Técnicas

| Decisão Técnica | Justificativa | Trade-off (Risco) |
| :--- | :--- | :--- |
| **Lag Features (T-1)** | **Anti-Leakage:** Garante que o modelo é preditivo, utilizando apenas dados disponíveis no momento da predição (início do ano). | **Dependência de Dados:** Requer um histórico limpo e consistente de pelo menos 2 anos para funcionar. |
| **Split Temporal** | **Validação Realista:** Simula o uso em produção, onde o modelo treinado no passado deve prever o futuro. | **Menor Volume de Treino:** Reduz o tamanho do conjunto de treino em comparação com um *split* aleatório. |
| **FastAPI + Singleton** | **Performance e Concorrência:** FastAPI oferece alta performance assíncrona. O *Singleton* (`ModelManager`) garante que o modelo seja carregado uma única vez, otimizando o uso de memória e reduzindo a latência de predição. | **Memória:** O modelo fica residente na memória do servidor, exigindo mais RAM. |
| **Evidently AI** | **Observabilidade MLOps:** Solução *open-source* para monitoramento de *Data Drift* e *Concept Drift*, essencial para a manutenção do modelo em produção. | **Infraestrutura:** Requer um *endpoint* dedicado (`/monitoring/dashboard`) e um mecanismo de persistência de logs (`prediction.jsonl`). |

## 8. Stack Tecnológica

| Categoria | Tecnologia | Uso |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.11+ | Desenvolvimento de todo o sistema. |
| **API** | FastAPI | Framework web de alta performance para o serviço de inferência. |
| **ML Core** | Scikit-learn | Treinamento do modelo (`RandomForestClassifier`) e pré-processamento (`Pipeline`, `ColumnTransformer`). |
| **Data** | Pandas, Joblib | Manipulação de dados e serialização/desserialização do modelo. |
| **Validação** | Pydantic | Definição de schemas de entrada (`StudentInput`, `Student`) e validação automática de dados. |
| **MLOps** | Evidently AI | Geração de relatórios de *Data Drift* em tempo real. |
| **Infraestrutura** | Docker, Docker Compose | Empacotamento e orquestração do ambiente de desenvolvimento/produção. |

## 9. Estrutura do Projeto

A estrutura de diretórios segue um padrão de projeto limpo e modular:

```
.
├── app/
│   ├── data/                   # Dados de entrada (Ex: PEDE_PASSOS_DATASET_FIAP.xlsx)
│   ├── models/                 # Modelos serializados (Ex: model_passos_magicos.joblib)
│   ├── src/                    # Código-fonte da aplicação
│   │   ├── api/                # Controladores (FastAPI)
│   │   ├── application/        # Lógica de Negócio (Services)
│   │   ├── config/             # Configurações globais (settings.py)
│   │   ├── domain/             # Modelos de Domínio (Pydantic)
│   │   ├── infrastructure/     # Implementações de Infraestrutura (ML, Data, Logging)
│   │   └── util/               # Utilitários (Ex: logger.py)
│   ├── main.py                 # Ponto de entrada da API
│   └── train.py                # Script de treinamento do modelo
├── tests/                      # Testes unitários e de integração
├── Dockerfile                  # Definição do ambiente Docker
├── docker-compose.yml          # Orquestração de serviços
└── requirements.txt            # Dependências do Python
```

## 10. API e Deployment

### 10.1. Endpoints

A API expõe dois endpoints principais para predição e um para observabilidade:

| Método | Path | Descrição | Audiência |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/predict/full` | Predição bruta. Requer todas as *features* (incluindo as *lag features*) no *payload*. | Desenvolvedores/Testes |
| `POST` | `/api/v1/predict/smart` | **Endpoint de Produção.** Requer apenas dados básicos do aluno. O sistema busca automaticamente o histórico (T-1) no `HistoricalRepository` para enriquecer o *payload*. | Sistemas Externos/Front-end |
| `GET` | `/api/v1/monitoring/dashboard` | Retorna o *dashboard* HTML do Evidently AI com a análise de *Data Drift*. | DevOps/MLOps |
| `GET` | `/health` | Checagem de saúde básica da API. | Infraestrutura/Load Balancer |

### 10.2. Exemplo de Uso (`/predict/smart`)

O endpoint `smart` é o recomendado para uso em produção, pois abstrai a complexidade do histórico.

**Payload de Entrada (Aluno Novo - RA 1500):**

```json
{
  "RA": "1500",
  "IDADE": 10,
  "ANO_INGRESSO": 2024,
  "GENERO": "Feminino",
  "TURMA": "1A",
  "INSTITUICAO_ENSINO": "MUNICIPAL",
  "FASE": "1A"
}
```

**Resposta de Saída (200 OK):**

```json
{
  "risk_probability": 0.4652,
  "risk_label": "BAIXO RISCO",
  "prediction": 0
}
```

### 10.3. Deployment (Docker)

O projeto é totalmente conteinerizado para garantir a portabilidade e a reprodutibilidade do ambiente.

1.  **Construção da Imagem:**
    ```bash
    docker build -t passos-magicos-api .
    ```
2.  **Execução (Com Docker Compose):**
    ```bash
    docker-compose up --build
    ```
    A API estará acessível em `http://localhost:8000`.

## 11. Testes e Qualidade

O projeto inclui uma suíte de testes unitários e de integração para garantir a qualidade do código e a integridade da lógica de ML.

| Componente Testado | Foco | Arquivos de Teste |
| :--- | :--- | :--- |
| **API** | Validação de *schemas* (Pydantic), status codes, e injeção de dependência. | `tests/api/` |
| **Domain** | Regras de validação dos modelos de domínio (`Student`, `StudentInput`). | `tests/domain/test_student.py` |
| **Infraestrutura** | Lógica de carregamento de dados (`DataLoader`), criação de *Lag Features* e *Quality Gate* do `MLPipeline`. | `tests/infrastructure/` |
| **Funcional** | Scripts em `scripts/` simulam requisições reais para validar o fluxo de ponta a ponta. | `scripts/funcional_real.py` |

## 12. Monitoramento e Observabilidade

A observabilidade é um pilar deste projeto MLOps, focada na detecção de desvios de dados (*Data Drift*).

### 12.1. Data Drift (Evidently AI)

O `MonitoringController` expõe um *dashboard* que compara o `reference_data.csv` (dados de treinamento) com o `prediction.jsonl` (dados de inferência em produção).

*   **Referência:** `reference_data.csv` (salvo após a promoção do modelo).
*   **Corrente:** Dados de *input* e *output* logados em `prediction.jsonl` pelo `PredictionLogger`.

### 12.2. Logging Estruturado

O `PredictionLogger` registra cada predição em formato **JSON Lines (JSONL)**, garantindo:

1.  **Atomicidade:** Escrita thread-safe (via `threading.Lock`) para ambientes concorrentes.
2.  **Estrutura:** O log inclui `timestamp`, `model_version`, `input_features` (as features usadas na predição) e `prediction_result`.
3.  **Rastreabilidade:** Uso de `prediction_id` e `correlation_id` para rastrear requisições.

## 13. Segurança e Confiabilidade

| Aspecto | Implementação |
| :--- | :--- |
| **Confiabilidade do Modelo** | **Quality Gate** (F1-Score > 95% do modelo atual) para evitar a promoção de modelos inferiores. |
| **Disponibilidade** | **Singleton** (`ModelManager`) e *health check* (`/health`) para garantir que o modelo esteja sempre pronto para inferência. |
| **Integridade de Dados** | **Pydantic** para validação de *schema* na entrada da API, rejeitando *payloads* malformados. |
| **Segurança (A Ser Implementado)** | **Falta de AuthN/AuthZ** é um risco conhecido. Recomenda-se a implementação de *API Keys* ou *tokens* JWT para proteger o endpoint de predição. |

## 14. Limitações Conhecidas

1.  **Feature Store Volátil:** O `HistoricalRepository` é um *Singleton* em memória. Em caso de reinicialização do contêiner, os dados históricos são recarregados do arquivo de referência, o que pode causar latência no *startup*.
2.  **Dependência de Arquivo:** O `DataLoader` é altamente acoplado ao formato e à estrutura do arquivo `PEDE_PASSOS_DATASET_FIAP.xlsx`. Qualquer alteração no *schema* do Excel pode quebrar o pipeline de treinamento.
3.  **Log de Produção:** O arquivo `prediction.jsonl` cresce indefinidamente. É necessária uma estratégia de rotação de logs (ex: Logrotate, ou envio para um *data sink* como Kafka/S3) para evitar o esgotamento do disco.

## 15. Possíveis Evoluções

O projeto está em um estado de **Produção Candidata (MLOps Nível 2)**. As próximas etapas de evolução incluem:

| Área | Melhoria Proposta | Impacto |
| :--- | :--- | :--- |
| **Infraestrutura** | Orquestração de *Retraining* (Airflow/Kubeflow) | Automação completa do ciclo de vida do ML. |
| **Feature Store** | Migração para Redis ou Feast | Persistência e escalabilidade do enriquecimento de *features* históricas. |
| **Segurança** | Implementação de AuthN/AuthZ na API | Proteção do endpoint de predição. |
| **Monitoramento** | Alerta de Drift (Slack/PagerDuty) | Notificação proativa quando o *Data Drift* ultrapassar um limite. |
| **Modelo** | Experimentação com modelos de *Deep Learning* (Ex: LSTMs) | Captura de padrões temporais mais complexos no histórico do aluno. |

## 16. Instruções de Execução

### 16.1. Pré-requisitos

*   Docker e Docker Compose instalados.
*   Python 3.11+ (para execução local).

### 16.2. Treinamento do Modelo (Offline)

O treinamento deve ser executado antes do *deployment* da API para gerar o modelo (`.joblib`), as métricas (`metrics.json`) e os dados de referência (`reference_data.csv`).

```bash
# 1. Navegue para o diretório da aplicação
cd project_repo/app

# 2. Execute o script de treinamento
python train.py
```

Se o *Quality Gate* for aprovado, os arquivos de produção serão atualizados em `app/models/`.

### 16.3. Execução da API (Online)

Utilize o Docker Compose para subir a API e o ambiente de forma isolada.

```bash
# 1. Navegue para o diretório raiz do projeto
cd project_repo

# 2. Suba os contêineres
docker-compose up --build
```

A API estará disponível em `http://localhost:8000`.

### 16.4. Simulação de Tráfego

Após a API estar rodando, utilize os scripts de simulação para gerar logs de predição e alimentar o *dashboard* de monitoramento.

```bash
# Em um novo terminal, na raiz do projeto:
cd project_repo/scripts

# Simula um fluxo contínuo de requisições com dados reais
python send_production_simulation.py
```

## 17. Conformidade com o Datathon

A solução atende aos requisitos de um projeto de Datathon de alto nível, com foco em MLOps e integridade preditiva.

| Requisito do Datathon | Implementação no Projeto |
| :--- | :--- |
| **Modelo Preditivo** | `RandomForestClassifier` treinado com *Lag Features* (T-1). |
| **Anti-Leakage** | **Split Temporal** e remoção de colunas proibidas (`COLUNAS_PROIBIDAS_NO_TREINO`). |
| **API de Inferência** | FastAPI com endpoint `/predict/smart` de baixa latência. |
| **Enriquecimento de Dados** | `HistoricalRepository` para busca automática de histórico (T-1). |
| **Monitoramento** | `MonitoringController` e `MonitoringService` com Evidently AI para *Data Drift*. |
| **Reprodutibilidade** | Dockerfile e `requirements.txt` para ambiente isolado. |

## 18. Uso Responsável e Ético

O modelo preditivo de risco é uma ferramenta de apoio, e não um oráculo.

*   **Transparência:** O modelo é baseado em *Random Forest*, que permite a extração de importância de *features* para explicar a predição.
*   **Viés e Equidade:** O *target* é baseado em métricas de desempenho e defasagem, que podem refletir vieses sistêmicos. O monitoramento de *drift* ajuda a identificar desvios na distribuição de *features* demográficas (ex: `GENERO`, `FASE`) que possam indicar *drift* de equidade.
*   **Intervenção Humana:** A decisão final de intervenção pedagógica deve ser sempre tomada por um profissional, utilizando a probabilidade de risco como um **sinal de alerta**, e não como uma sentença.

## 19. Considerações Finais

Este projeto demonstra a maturidade técnica necessária para transicionar um modelo de ML de um ambiente de pesquisa para um ambiente de produção. A ênfase na prevenção de *data leakage* e na implementação de práticas MLOps (Quality Gate, Monitoramento, Logging Estruturado) garante que a solução seja **confiável, sustentável e eticamente responsável** no apoio à missão da ONG Passos Mágicos.

## 20. Licença e Contribuição

Este projeto está licenciado sob a Licença MIT.

---
