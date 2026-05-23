# RAG Project — Sistema de Embeddings com Banco Vetorial e Integração IA

Sistema completo de RAG (Retrieval-Augmented Generation) que ingere PDFs/livros, gera embeddings semânticos, armazena no Qdrant e responde perguntas com contexto recuperado por similaridade.

---

## Estrutura

```
rag-project/
├── src/
│   ├── app.py           # Interface web Flask
│   ├── ingest.py        # Extrai PDF, divide em trechos, gera embeddings e salva no Qdrant
│   ├── query.py         # Busca contexto no Qdrant e gera prompt para IA
│   └── templates/
│       └── index.html   # Frontend da interface web
├── scripts/
│   ├── start.sh                 # Inicialização completa (venv + Qdrant + testes + web)
│   ├── stop.sh                  # Encerra servidor web e Qdrant
│   ├── run_web.sh               # Inicia apenas a interface web Flask
│   ├── install_dependencies.sh  # Instala dependências Python
│   ├── start_qdrant.sh          # Sobe o Qdrant via Docker
│   ├── run_ingest.sh            # Executa a ingestão de um PDF
│   └── run_query.sh             # Inicia o loop de perguntas e respostas
├── tests/
│   └── test_rag.py      # Testes unitários dos componentes principais
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.8+
- Docker (para rodar o Qdrant)

---

## Início Rápido

```bash
# Inicializa tudo: venv, dependências, Qdrant, testes e interface web
bash scripts/start.sh
```

Acesse a interface em **http://localhost:5000** e o painel do Qdrant em **http://localhost:6333/dashboard**.

Para encerrar:

```bash
bash scripts/stop.sh
```

---

## Instalação Manual

```bash
# 1. Instale as dependências Python
bash scripts/install_dependencies.sh

# 2. Suba o banco vetorial Qdrant
bash scripts/start_qdrant.sh
```

---

## Uso

### Interface Web

```bash
# Via script de inicialização completa (recomendado)
bash scripts/start.sh

# Ou apenas a interface web (Qdrant já deve estar rodando)
bash scripts/run_web.sh
# Com porta customizada:
bash scripts/run_web.sh 8080
```

### Ingerir um PDF

```bash
bash scripts/run_ingest.sh caminho/para/livro.pdf
# Ou com parâmetros customizados:
bash scripts/run_ingest.sh livro.pdf minha_collection 500
```

### Fazer perguntas (CLI)

```bash
bash scripts/run_query.sh
# Ou com parâmetros customizados:
bash scripts/run_query.sh minha_collection 10
```

---

## Integrar com um modelo de IA

Em `src/query.py`, substitua a função `default_ai_model` pela integração com o modelo de sua escolha:

```python
# Exemplo com OpenAI
import openai

def openai_model(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

Passe a função ao chamar `query_loop`:

```python
query_loop(collection_name, top_k, ai_model_fn=openai_model)
```

---

## Testes

```bash
python -m pytest tests/test_rag.py -v
```

---

## Checklist de Qualidade

- [ ] Qdrant está rodando (`docker ps`)
- [ ] Dependências instaladas (`pip list`)
- [ ] PDF ingerido sem erros
- [ ] Embeddings com dimensão 384
- [ ] Busca retorna trechos relevantes
- [ ] Prompt contém contexto e pergunta
- [ ] Modelo de IA configurado e respondendo
- [ ] Todos os testes passando
