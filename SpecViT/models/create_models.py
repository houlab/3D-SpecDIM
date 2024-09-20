import importlib

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None

    if model_name == 'cnn':
        target_model_name = 'CNN'
    elif model_name == 'ViT' or model_name == 'ViT_skip':
        target_model_name = 'ViT'
    elif model_name == 'DViT':
        target_model_name = 'DViT'
    elif model_name == 'DAR_ViT':
        target_model_name = 'DAR_ViT'
    elif model_name == 'MDD_ViT':
        target_model_name = 'MDD_ViT'
    elif model_name == 'L_DViT':
        target_model_name = 'L_DViT'
    else:
        target_model_name = ''
    
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    print("model [%s] was created" % opt.model)
    return model
