# Technology Stack Reference

## Backend Technologies

### Python Frameworks

**Flask**
- Lightweight WSGI framework
- Minimal boilerplate, maximum flexibility
- Great for: APIs, microservices, smaller applications
- Extensions: Flask-SQLAlchemy, Flask-Login, Flask-CORS

**FastAPI**
- Modern async framework built on Starlette and Pydantic
- Automatic API documentation (Swagger/ReDoc)
- Type hints for validation and serialization
- Great for: High-performance APIs, ML model serving, async operations

**Django**
- Full-featured MTV framework
- Built-in admin, ORM, authentication
- Great for: Complex applications, rapid development, traditional web apps

### Node.js Frameworks

**Express**
- Minimalist, unopinionated framework
- Large ecosystem of middleware
- Great for: Simple APIs, real-time apps, prototypes

**NestJS**
- Enterprise TypeScript framework
- Dependency injection, modular architecture
- Great for: Large teams, complex backends, microservices

## Frontend Technologies

### React Ecosystem

**Core Libraries:**
- React 18+ with Suspense and Concurrent features
- React Router for navigation
- React Query for server state management

**State Management:**
- Zustand: Simple, lightweight (preferred for most cases)
- Redux Toolkit: Complex state, time-travel debugging
- Jotai/Recoil: Atomic state management

**Frameworks:**
- Next.js: SSR, SSG, API routes, image optimization
- Vite: Fast dev server, optimized builds

**Styling:**
- Tailwind CSS: Utility-first, rapid development
- Styled Components: CSS-in-JS
- CSS Modules: Scoped styles

## Databases

### Relational
- **PostgreSQL**: Production default, JSONB support, full-text search
- **MySQL**: Wide adoption, good for read-heavy workloads

### NoSQL
- **MongoDB**: Document store, flexible schemas
- **Redis**: In-memory, caching, session storage, pub/sub

### Vector Databases
- **Pinecone**: Managed, scalable, easy to use
- **Weaviate**: Open-source, GraphQL API
- **Chroma**: Lightweight, embeddable
- **FAISS**: Facebook's vector search library

## AI/ML Technologies

### LLM Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo, embeddings
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro, PaLM 2
- **Open-source**: Llama 2, Mistral, Mixtral

### ML Frameworks
- **PyTorch**: Research and production, dynamic graphs
- **TensorFlow**: Production ML, TensorFlow Serving
- **Hugging Face Transformers**: Pre-trained models, fine-tuning
- **Scikit-learn**: Traditional ML algorithms

### LLM Tools
- **LangChain**: LLM orchestration, chains, agents
- **LlamaIndex**: Data indexing and retrieval
- **Haystack**: NLP pipelines
- **Semantic Kernel**: Microsoft's LLM SDK

### MLOps
- **Weights & Biases**: Experiment tracking
- **MLflow**: Model registry and tracking
- **DVC**: Data version control
- **Kubeflow**: ML on Kubernetes

## Cloud Platforms

### AWS
- **Compute**: EC2, Lambda, ECS, EKS
- **Storage**: S3, EBS, EFS
- **Database**: RDS, DynamoDB, ElastiCache
- **AI/ML**: SageMaker, Bedrock
- **Monitoring**: CloudWatch, X-Ray

### GCP
- **Compute**: Compute Engine, Cloud Run, GKE
- **Storage**: Cloud Storage, Persistent Disk
- **Database**: Cloud SQL, Firestore, Memorystore
- **AI/ML**: Vertex AI, AI Platform
- **Monitoring**: Cloud Monitoring, Cloud Trace

### Azure
- **Compute**: Virtual Machines, App Service, AKS
- **Storage**: Blob Storage, Disk Storage
- **Database**: Azure SQL, Cosmos DB, Azure Cache
- **AI/ML**: Azure ML, Azure OpenAI Service
- **Monitoring**: Azure Monitor, Application Insights

## DevOps Tools

### Containerization
- **Docker**: Container runtime
- **Docker Compose**: Multi-container orchestration
- **Kubernetes**: Production container orchestration

### CI/CD
- **GitHub Actions**: Integrated with GitHub
- **GitLab CI**: Built into GitLab
- **Jenkins**: Self-hosted, highly customizable
- **CircleCI**: Cloud-based, parallelization

### Monitoring
- **Prometheus + Grafana**: Metrics and visualization
- **ELK Stack**: Elasticsearch, Logstash, Kibana for logs
- **Datadog**: All-in-one monitoring platform
- **New Relic**: APM and infrastructure monitoring

### Infrastructure as Code
- **Terraform**: Multi-cloud, declarative
- **AWS CloudFormation**: AWS-specific
- **Pulumi**: Programming language-based IaC

## Development Tools

### Version Control
- Git with conventional commits
- GitHub/GitLab/Bitbucket

### API Development
- Postman for API testing
- Insomnia for REST/GraphQL
- OpenAPI/Swagger for documentation

### Code Quality
- **Python**: pylint, black, mypy, ruff
- **JavaScript/TypeScript**: ESLint, Prettier
- **Pre-commit hooks**: husky, lint-staged

### Testing
- **Python**: pytest, unittest, coverage
- **JavaScript**: Jest, Vitest, Testing Library
- **E2E**: Playwright, Cypress
- **Load testing**: Locust, k6, Apache JMeter
