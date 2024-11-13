'''this will need adaptations to the IBM HPC environment, without it I can only
get this so far --> we need their SDK 
'''

import os
import yaml 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#need to import service for IBM HPC 

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def init_ibm_client(api_key):
    authenticator = IAMAuthenticator(api_key)
    client = APIClient({
        'url': 'https://us-south.ml.cloud.ibm.com',     #this will probably change based on region
        'authenticator': authenticator
    })
    return client


def submit_ibm_hpc_job(client, job_name, script_path, parameters=None):
    job_payload = {
        "job_name": job_name,
        "script_path": script_path,
        "parameters": parameters or {}
    }

    try: 
        deploy_details = client.deployments.create(paylaod=job_payload)
        job_id = deploy_details['metadata']['guid']
        print(f"Job: {job_name} submitted with ID {job_id}.")
        return job_id
    except Exception as e:
        print(f"Error submitting job: {job_name}: {e}")
        return None
    

def monitor_ibm_job(client, job_id):
    print(f"Monitoring job {job_id}")
    while True:
        try:
            status = client.deployment.get_details(job_id)['entity']['status']['state']
            print(f"Job {job_id} status:{status}")
            if status in ('completed', ('failed')):
                break
        except Exception as e:
            print(f"Error monitoring job: {job_name}: {e}")
            break
    return status




