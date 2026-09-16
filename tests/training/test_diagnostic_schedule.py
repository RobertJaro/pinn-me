from types import SimpleNamespace
from unittest.mock import Mock

from prom3theus.application.joint_training import JointTrainingDiagnostics


def test_diagnostics_skip_startup_and_keep_interval_and_final_evaluation():
    callback = JointTrainingDiagnostics(None)
    callback._evaluate = Mock()
    trainer = SimpleNamespace(global_step=0)
    module = SimpleNamespace(settings=SimpleNamespace(validation_every_n_steps=100))
    callback.on_train_start(trainer, module)
    callback._evaluate.assert_not_called()
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback._evaluate.assert_not_called()
    trainer.global_step = 100
    callback.on_train_batch_end(trainer, module, None, None, 99)
    callback._evaluate.assert_called_once_with(trainer, module)
    callback.on_train_end(trainer, module)
    assert callback._evaluate.call_count == 2
