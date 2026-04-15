import types

from agedi.data.callbacks import TrainingPhase


def test_prepare_epoch_increments_counter_without_phase_change():
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[2, 2])
    trainer = types.SimpleNamespace(
        current_epoch=0,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: None),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 0
    assert callback.epoch_counter == 1


def test_prepare_epoch_advances_phase_and_resets_counter():
    called = []
    callback = TrainingPhase(n_phases=3, epochs_per_phase=[1, 1, 1])
    callback.epoch_counter = 1
    trainer = types.SimpleNamespace(
        current_epoch=1,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: called.append(phase)),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 1
    assert callback.epoch_counter == 0
    assert called == [1]


def test_prepare_epoch_noop_in_last_phase():
    called = []
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[1, 1])
    callback.current_phase = 1
    callback.epoch_counter = 10
    trainer = types.SimpleNamespace(
        current_epoch=10,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: called.append(phase)),
    )

    callback._prepare_epoch(trainer, model=None)

    assert callback.current_phase == 1
    assert callback.epoch_counter == 10
    assert called == []


def test_on_validation_end_delegates_to_prepare_epoch():
    callback = TrainingPhase(n_phases=2, epochs_per_phase=[1, 1])
    trainer = types.SimpleNamespace(
        current_epoch=0,
        datamodule=types.SimpleNamespace(set_phase=lambda phase: None),
    )

    callback.on_validation_end(trainer, model=None)

    assert callback.epoch_counter == 1

