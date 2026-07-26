# Streamlit dashboard image. Used by Render (see render.yaml); runs anywhere Docker does.
#
# The JAK models, datasets and conformal quantiles are committed under assets/,
# so this image starts a screen without retraining — see README "Deploying".
FROM python:3.11-slim

WORKDIR /app

# RDKit's molecule drawing needs X libraries. packages.txt is the single source of
# truth for them — Streamlit Community Cloud reads that same file. Comments (#) and
# blank lines are stripped since Community Cloud's parser accepts them but a plain
# `tr` here would otherwise pass "#" and comment words to apt-get as package names.
COPY packages.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       $(grep -v '^#' packages.txt | grep -v '^\s*$') \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The Colab handoff pins the deep dive to the commit the contract was exported
# from, and this image has neither git nor .git — so pass the commit in, or the
# dashboard has no commit to pin and drops the link:
#
#   docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
#
# Render needs no build arg: it injects RENDER_GIT_COMMIT / RENDER_GIT_REPO_SLUG
# at runtime and src/loop_contract.py reads those too.
ARG GIT_COMMIT=unknown
ENV CHEM_PREDICT_COMMIT=$GIT_COMMIT
ENV CHEM_PREDICT_REPO=zqvo04/chem-predict-dashboard

# Render injects $PORT; default to Streamlit's own for local `docker run`.
ENV PORT=8501
EXPOSE 8501

CMD streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
