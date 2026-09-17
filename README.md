# Agente Normativo PyC

Prototipo RAG en Python y Streamlit para consultar documentación normativa pública de la CMF.

## Arquitectura

1. PyPDF extrae texto y conserva documento/página.
2. El contenido se divide en fragmentos.
3. TF-IDF convierte fragmentos y preguntas en vectores.
4. Se recuperan los fragmentos más relevantes.
5. Gemini redacta la respuesta exclusivamente con ese contexto.
6. La interfaz muestra las fuentes recuperadas.

La capa de recuperación y la capa LLM están separadas. En una evolución empresarial se puede reemplazar TF-IDF por embeddings + Azure AI Search y Gemini por Azure OpenAI.

## Ejecución local

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Sin clave API, la aplicación funciona en modo demostración extractiva. Para habilitar respuestas redactadas, copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y agrega una clave válida.

## Publicación en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir el contenido de esta carpeta a la raíz del repositorio.
3. Entrar a `share.streamlit.io` con GitHub.
4. Seleccionar el repositorio y `app.py`.
5. En **Advanced settings / Secrets**, agregar:

```toml
GEMINI_API_KEY = "tu_clave"
```

6. Desplegar y probar la URL generada.

Nunca publiques `.streamlit/secrets.toml` ni una clave dentro del código.

## Preguntas de prueba

- ¿En qué moneda deben informarse los montos del sistema contable?
- ¿Cuál es la periodicidad y el plazo del archivo MB1?
- ¿Qué personas deben incluirse en el archivo D03?
- ¿Cómo se convierten a pesos las operaciones en moneda extranjera?
- ¿Cómo se mantiene actualizado el Manual del Sistema de Información?

## Alcance

Es una demostración pública con documentos públicos. No debe utilizarse con información confidencial, credenciales, datos personales ni documentos de clientes sin incorporar autenticación, autorización y una infraestructura empresarial.
