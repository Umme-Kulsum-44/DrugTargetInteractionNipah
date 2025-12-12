FROM condaforge/mambaforge:latest

# Set work directory
WORKDIR /app

# Copy environment and code
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt
COPY . /app

# Create conda environment
RUN mamba env create -f /app/environment.yml -n appenv -y

# Add conda environment to PATH
ENV PATH /opt/conda/envs/appenv/bin:$PATH

# Install pip-only dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt

# Expose Render port
EXPOSE 10000

# Run app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2"]
