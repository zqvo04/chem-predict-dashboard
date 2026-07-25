# Streamlit dashboard image. Used by Render (see render.yaml); runs anywhere Docker does.
#
# The JAK models, datasets and conformal quantiles are committed under assets/,
# so this image starts a screen without retraining — see README "Deploying".
FROM python:3.11-slim

WORKDIR /app

# RDKit's molecule drawing needs X libraries. packages.txt is the single source of
# truth for them — Streamlit Community Cloud reads that same file.
COPY packages.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends $(tr '\n' ' ' < packages.txt) \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects $PORT; default to Streamlit's own for local `docker run`.
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
