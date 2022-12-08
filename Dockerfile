# getting base image of spark
FROM datamechanics/spark:3.2.1-latest
ENV PYSPARK_MAJOR_PYTHON_VERSION=3
WORKDIR /Users/dheer/CS643_Wine_Prediction

RUN conda install numpy
RUN conda install pandas

COPY dp796-ModelValidation.py .  
ADD ValidationDataset.csv .
ADD trainedmdl ./model/