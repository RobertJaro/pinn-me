def test_preparation_fingerprint_ignores_trainer_and_diagnostic_edits(tmp_path, monkeypatch):
    from prom3theus.observations import identity

    root = tmp_path / 'package'
    for directory in ('observations', 'training', 'diagnostics'):
        (root / directory).mkdir(parents=True)
    observation = root / 'observations' / 'store.py'
    trainer = root / 'training' / 'joint.py'
    diagnostic = root / 'diagnostics' / 'plot.py'
    observation.write_text('preparation-v1')
    trainer.write_text('trainer-v1')
    diagnostic.write_text('plot-v1')
    monkeypatch.setattr(identity, '__file__', str(root / 'observations' / 'identity.py'))
    first = identity.implementation_signature()
    trainer.write_text('trainer-v2')
    diagnostic.write_text('plot-v2')
    assert identity.implementation_signature() == first
    observation.write_text('preparation-v2')
    assert identity.implementation_signature() != first
