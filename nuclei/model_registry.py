from yolov8 import Yolov8SegmentationONNX, Yolov8Generator

class ModelRegistry:
    __entry__ = ['model', 'generator']
    def __init__(self):
        self._registry = {k: {} for k in self.__entry__}
    
    def register(self, name, entry, cls):
        assert entry in self.__entry__
        self._registry[entry][name] = cls

    def get_model(self, name):
        return self._registry['model'].get(name)
    
    def get_generator(self, name):
        return self._registry['generator'].get(name)
    
    def info(self):
        return self._registry
        

MODEL_REGISTRY = ModelRegistry()

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY.registry(name, 'model', cls)
        return cls
    return decorator

def register_generator(name):
    def decorator(cls):
        MODEL_REGISTRY.registry(name, 'generator', cls)
        return cls
    return decorator


# MODEL_REGISTRY.register("HDYolo", "model", Yolov8SegmentationONNX)
# MODEL_REGISTRY.register("HDYolo", "generator", Yolov8Generator)
MODEL_REGISTRY.register("yolov8-lung", "model", Yolov8SegmentationONNX)
MODEL_REGISTRY.register("yolov8-lung", "generator", Yolov8Generator)
