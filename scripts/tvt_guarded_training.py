"""Use clean forecast targets without changing frozen historical training code."""
from src.train.trainer import Trainer


class CleanForecastTargetTrainer(Trainer):
    def _step(self, x, epoch):
        clean = x.to(self.device, non_blocking=True)

        def clean_target(module, args, output):
            forecast = output.get('lstm_forecast')
            if forecast is None:
                raise RuntimeError('Expected the registered auxiliary forecast')
            return {**output, 'lstm_target': clean[:, -forecast.shape[1]:, :]}

        handle = self.model.register_forward_hook(clean_target)
        try:
            return super()._step(clean, epoch)
        finally:
            handle.remove()
