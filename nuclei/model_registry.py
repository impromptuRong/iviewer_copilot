from yolov8 import Yolov8SegmentationONNX, Yolov8Generator, Yolov8Config

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

    def load_service(self, config, test=False):
        service = self.get_model(config.server)(
            config, device=config.device
        )
        if test:
            service.test_run()

        return service

    def info(self):
        return self._registry
        

MODEL_REGISTRY = ModelRegistry()
# MODEL_REGISTRY.register("HDYolo", "model", Yolov8SegmentationONNX)
# MODEL_REGISTRY.register("HDYolo", "generator", Yolov8Generator)
MODEL_REGISTRY.register("yolov8", "model", Yolov8SegmentationONNX)
MODEL_REGISTRY.register("yolov8", "generator", Yolov8Generator)


AGENT_CONFIGS = {
    'yolov8-lung': Yolov8Config(
        model_path = "./ckpts/nuclei-yolov8-lung/best.onnx",
    ),
    'yolov8-colon': Yolov8Config(
        model_path = "./ckpts/nuclei-yolov8-colon/best.onnx",
    ),
}
