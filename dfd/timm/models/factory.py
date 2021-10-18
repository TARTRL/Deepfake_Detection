from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint

from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint


def create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        ignore_keys=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])

    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=False, ignore_keys=ignore_keys)

    return model


def create_deepfake_model(
        model_name,
        pretrained=False,
        num_classes=2,
        in_chans=3,
        checkpoint_path='',
        ignore_keys=None,
        strict=False,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])

    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=strict, ignore_keys=ignore_keys)

    return model


def create_deepfake_model_v3(
        model_name,
        pretrained=False,
        num_classes=2,
        in_chans=3,
        checkpoint_path='',
        ignore_keys=None,
        strict=False,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    assert model_name == 'efficientnet_deepfake_v3', '{} != efficientnet_deepfake_v3'.format(model_name)

    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    assert is_efficientnet

    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=strict, ignore_keys=ignore_keys)

    return model


def create_deepfake_model_v4(
        model_name,
        pretrained=False,
        num_classes=2,
        in_chans=3,
        checkpoint_path='',
        ignore_keys=None,
        strict=False,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    assert model_name == 'efficientnet_deepfake_v4', '{} != efficientnet_deepfake_v4'.format(model_name)

    margs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    assert is_efficientnet

    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    # kwargs['stem_size'] = 32
    # print(margs,kwargs)
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**margs, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=strict, ignore_keys=ignore_keys)

    return model
