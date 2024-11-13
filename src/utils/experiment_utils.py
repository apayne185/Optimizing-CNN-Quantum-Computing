'''this needs to be adapted to IBM hpc'''

import os
import yaml
import logging
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_hpc_sdk import HPCClient 
from datetime import datetime



with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_ibm_client(api_key=None):
    try:
        if api_key:
            authenticator = IAMAuthenticator(api_key)
        else:
            authenticator = IAMAuthenticator(config['api_key'])  

        client = HPCClient({
            'url': config['hpc_url'], 
            'authenticator': authenticator
        })
        logger.info("IBM HPC client init successfully.")
        return client
    except Exception as e:
        logger.error(f"Error init IBM HPC client: {e}")
        return None


def submit_ibm_job(client, job_name, script_path, parameters=None):
    try:
        job_payload = {
            "job_name": job_name,
            "script_path": script_path,
            "parameters": parameters or {}
        }
        job_details = client.submit_job(payload=job_payload)  # adapt for IBM HPC
        job_id = job_details['metadata']['guid']
        logger.info(f"Job '{job_name}' submitted successfully with ID {job_id}.")
        return job_id
    
    except Exception as e:
        logger.error(f"Error submitting job '{job_name}': {e}")
        return None



def monitor_ibm_hpc_job(client, job_id):
    try:
        logger.info(f"Monitoring IBM HPC job {job_id}")
        while True:
            status = client.get_job_status(job_id)
            logger.info(f"Job {job_id} status: {status}")
            
            if status in ('completed', 'failed'):
                logger.info(f"Job {job_id} has {status}.")
                break
    except Exception as e:
        logger.error(f"Error monitoring job {job_id}: {e}")
    return status


def log_experiment_start(experiment_name, output_dir='./results/logs/'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")
    
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.info(f"Experiment: '{experiment_name}' started: {timestamp}.")
    return log_file



def log_experiment_results(log_file, results):
    try:
        with open(log_file, 'a') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
            logger.info("Results logged successfully.")
    except Exception as e:
        logger.error(f"Error logging results: {e}")



def log_training_metrics(epoch, logs, model_name="CNN"):
    try:
        logger.info(f"Epoch {epoch} - {model_name} Training: Loss ={logs['loss']}, Accuracy ={logs['accuracy']}")
    except KeyError as e:
        logger.error(f"Missing key in training logs: {e}")



def save_model_checkpoint(model, model_name, output_dir="./results/model_outputs"):
    try:
        model_path = os.path.join(output_dir, f"{model_name}_best_model.h5")
        model.save(model_path)
        logger.info(f"Model saved: {model_path}.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
