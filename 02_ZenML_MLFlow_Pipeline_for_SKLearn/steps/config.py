from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model configuration
    """
    
    model_name: str = "linear_regression"
    fine_tuning: bool = False