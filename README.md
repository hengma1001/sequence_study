# Protein Mutation Study



## Run AlphaFold for sequence generation 
To generate mutated protein sequence and structure, 


```
bash run_alpha.sh -f ${input_fasta} -o ${output_path} -a ${gpu_id} 
```

This github simplifies the setup for Alphafold particularly for lambda machine at ANL. Contact the author 
for implementations elsewhere.  


## Setting up
The script uses `singularity` image base on the AlphaFold docker, which can be built with 
the following steps. 

1. Build the docker image with 
    ```
    sudo docker build -f docker/Dockerfile -t alphafold .
    ```
2. Port the docker image to singularity. 
    ```
    sudo singularity  build alphafold.sif docker-daemon://alphafold:latest
    ```
