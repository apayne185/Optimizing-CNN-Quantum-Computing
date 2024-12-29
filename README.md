Notes:

install requirements.txt

**if you are on windows, follow these instructions:
1. open powershell as admin
2. wsl -- install
3. sudo apt update
4. sudo apt update -y python3 python3-pip
5. pip install tensorflow-quantum 

tensorflow-quantum is not compatible with windows, so we need to use WSL (windows subsystem for linux) for a linux environment to install 


Scripts
- ibm_backend.py : handles Q and HPC be logic, interacts with APIs and databases
    - needs functions to set up connections to IBMQ be, submit quantum tasks to be, and to track job progress and get results 
- imb_submit.py : contains code for jobs submission to IBM HPC 
    - needs functions for submitting the cnn and preprocessing jobs, etc

Results
- logs/ --> contains logs of experiments, training, runtime (tracks training/testing)
    - project_log.log --> consolidates logs from different parts of the project
    - experiments_logs/ --> we can make this subdirectory if we need it for specific logs 

- model_output/ --> saves serialized trained model weights/predictions/results from evaluation. likely will have the files:
    - model_{epoch}.h5  --> model checkpoints
    - metrics.json --> sumamry of model performance metrics for each run

src/Utils 
- data_preprocess.py --> 



https://www.tensorflow.org/quantum/tutorials/qcnn

https://www.geeksforgeeks.org/introduction-to-grovers-algorithm/
