from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    """
    1.) onnect to zenml first. e.g.
    > zenml connect --url http://192.168.2.110:8888 --username admin --password zenml
    2.) Make sure that ZenML SKLearn and MLFlow integration are installed
    > zenml integration install sklearn
    > zenml integration install mlflow
    3.) register the MLFlow tracker
    > zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    > zenml model-deployer register mlflow --flavor=mlflow
    > zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
    4. The MLFlow data is placed in /home/<user>/.config/zenml/local_stores/<ID>/mlruns
    > mlflow ui --backend-store-uri file:/home/<user>/.config/zenml/local_stores/<ID>/mlruns
    """

    # start pipeline
    training_pipeline(data_path='datasets/olist_complete_dataset.csv')