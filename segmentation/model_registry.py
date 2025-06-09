from sam2 import SAM2Segmentation, SAM2Config

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

    def load_service(self, config, device='cpu', test=False):
        service = self.get_model(config.server)(
            config, device=device,
        )
        if test:
            try:
                service.test_run()
            except Exception as e:
                print(f"Test run warning:{str(e)} (proceeding anyway)")

        return service

    def info(self):
        return self._registry
        

MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("sam2", "model", SAM2Segmentation)

AGENT_CONFIGS = {
    'sam2-b': SAM2Config(
        model_path = './ckpts/ultralytics/sam2_b.pt',
        default_input_size = (512, 512),
    ),
    'sam2-s': SAM2Config(
        model_path = './ckpts/ultralytics/sam2_s.pt',
        default_input_size = (512, 512),
    ),
}
