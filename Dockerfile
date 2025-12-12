# Dockerfile - builds a container with RDKit & your Flask app
FROM condaforge/mambaforge:latest

# Set working dir
WORKDIR /app

# Copy environment and requirements (if present)
COPY environment.yml /app/environment.yml
# If you have a pip requirements.txt, copy it too
COPY requirements.txt /app/requirements.txt
COPY . /app

# Create the conda env named 'appenv' from environment.yml
# -y to avoid prompt, --no-default-packages for speed
RUN mamba env create -f /app/environment.yml -n appenv -y \
  && echo "conda activate appenv" >> /etc/profile.d/conda.sh \
  && /opt/conda/bin/conda clean -afy

# Make conda env the default PATH
ENV PATH /opt/conda/envs/appenv/bin:$PATH

# If you still need pip-only packages listed in requirements.txt:
RUN if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

# Expose port used by Render (we’ll use 10000 as you did)
EXPOSE 10000

# Ensure Flask app is found as module 'app' and exposes `app` Flask object.
# Run with gunicorn binding to 0.0.0.0:10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2"]
