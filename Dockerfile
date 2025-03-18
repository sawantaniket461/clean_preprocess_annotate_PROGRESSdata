FROM python:3.12-slim

# Install R and R packages (if needed - keep if your Python script uses R)
RUN apt-get update && apt-get install -y r-base
RUN R -e "install.packages(c('dplyr', 'ggplot2', 'tidyr'), dependencies=TRUE)"

WORKDIR /app

# Create the /app/bin directory (important for consistent pathing)
RUN mkdir -p /app/bin

# COPY ONLY the Python script! (Crucial!)
COPY bin/CleanOneLineDataTable_clean.py /app/bin/CleanOneLineDataTable_clean.py
COPY bin /app/bin
COPY inFiles /app/inFiles
COPY outFiles /app/outFiles

# Copy requirements.txt and install dependencies (if you have a requirements file)
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Make the script executable (inside the container)
RUN chmod +x /app/bin/CleanOneLineDataTable_clean.py
