# Agente Normativo PyC

Prototipo RAG en Python y Streamlit para consultar documentación normativa pública de la CMF.

## Arquitectura

1. PyPDF extrae texto y conserva documento/página.
2. El contenido se divide en fragmentos.
3. Un índice híbrido combina palabras, caracteres y códigos normativos exactos.
4. Se recuperan candidatos, se reordenan y se selecciona la mejor evidencia.
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

Sin clave API, la aplicación funciona en modo demostración extractiva. Para habilitar respuestas redactadas y la administración, copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y configura ambas claves.

```toml
GEMINI_API_KEY = "tu_clave_de_gemini"
ADMIN_PASSWORD = "una_clave_administrativa_distinta"
```

## Administración de conocimiento

La pestaña **Administración** permite:

1. Autenticarse con una clave administrativa.
2. Cargar uno o varios PDF.
3. Completar organismo, área, tipo, versión, vigencia, estado y responsable.
4. Validar páginas con texto, duplicados, tamaño y códigos normativos detectados.
5. Incorporar los documentos y reconstruir la base.
6. Descargar un ZIP de respaldo para hacer permanente la actualización en GitHub.

La carga hecha dentro de Streamlit Community Cloud puede perderse cuando el servidor
se reinicia. Para una actualización permanente, descargue el paquete generado y suba
los PDF y `knowledge_metadata.json` al repositorio. En una versión empresarial, los
documentos deben persistirse en SharePoint o Azure Blob Storage.

## Publicación en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir el contenido de esta carpeta a la raíz del repositorio.
3. Entrar a `share.streamlit.io` con GitHub.
4. Seleccionar el repositorio y `app.py`.
5. En **Advanced settings / Secrets**, agregar:

```toml
GEMINI_API_KEY = "tu_clave"
ADMIN_PASSWORD = "tu_clave_administrativa"
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
